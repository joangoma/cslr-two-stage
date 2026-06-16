"""
cascade_evaluate.py
-------------------
Evaluates the full Video → Gloss (CTC) → Catalan Text (M2M100) cascade.

Two modes are always run back-to-back:

  CASCADE : video → predicted glosses → predicted text   (real system)
  ORACLE  : reference glosses  → predicted text          (stage-1 upper bound)

Comparing CASCADE vs ORACLE isolates exactly how much the gloss errors
(WER) hurt the final translation quality.

Metrics per mode
  BLEU        sacrebleu
  chrF        character n-gram F-score
  BERTScore   F1 with xlm-roberta-large (covers Catalan natively)
  COMET-Kiwi  QE-based metric, no source needed — same protocol used in
              audio/speech-translation papers (SeamlessM4T, mSLAM, etc.)
              Model: Unbabel/wmt23-cometkiwi-da-xl

Intermediate : gloss WER (stage 1 only)

Install (once on your HPC env):
    pip install unbabel-comet==2.2.7 bert-score==0.3.13 evaluate sacrebleu

Usage:
    python cascade_evaluate.py \\
        --v2g_checkpoint  /path/to/best_model.pt \\
        --g2t_model_dir   /path/to/gloss_to_text/final_model \\
        --data_file       /path/to/val.json \\
        --translation_field translation \\
        [--batch_size_v2g 1] [--batch_size_g2t 16] [--num_beams 5] [--fp16]
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import evaluate as hf_evaluate

# ---------------------------------------------------------------------------
# Optional heavy metrics — imported lazily so the script still runs if one
# of the packages isn't installed.
# ---------------------------------------------------------------------------
try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    warnings.warn(
        "bert-score not installed. BERTScore will be skipped.\n"
        "  pip install bert-score==0.3.13",
        stacklevel=1,
    )

try:
    from comet import load_from_checkpoint, download_model
    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False
    warnings.warn(
        "unbabel-comet not installed. COMET-Kiwi will be skipped.\n"
        "  pip install unbabel-comet==2.2.7",
        stacklevel=1,
    )

# Point to the sibling experiment folders so local modules resolve
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../13_video_to_glosses/video_to_glosses/"))
sys.path.insert(0, os.path.join(_HERE, "../12_glosses_to_text"))

from dataset import SignVideoDataset, SignLanguageCollate
from model   import VideoToGlossModel
from metrics import batch_ctc_greedy_decode


# ---------------------------------------------------------------------------
# COMET-Kiwi model name
# wmt23-cometkiwi-da-xl is the reference-based QE model used in speech-
# translation evaluation (SeamlessM4T, NLLB, etc.).  It takes
# (hypothesis, reference) with no source — correct for our cascade
# since our "source" is a video, not text.
# ---------------------------------------------------------------------------
COMETKIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"

# BERTScore backbone: xlm-roberta-large covers Catalan and is the standard
# choice for multilingual / low-resource evaluation in the NLP literature.
BERTSCORE_MODEL = "xlm-roberta-large"


# ---------------------------------------------------------------------------
# Dataset / collator extensions
# ---------------------------------------------------------------------------

class CascadeEvalDataset(SignVideoDataset):
    """
    Thin wrapper around SignVideoDataset that also surfaces:
      - the reference Catalan translation (from val.json)
      - the original sample index (so we can trace skipped samples)
    """

    def __init__(self, *args, translation_field: str = "translation", **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_field = translation_field

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if result is None:
            return None
        result["translation"] = self.data[idx].get(self.translation_field, "")
        result["sample_id"]   = idx
        return result


class CascadeCollate(SignLanguageCollate):
    """Extends SignLanguageCollate to also pack translations and sample ids."""

    def __call__(self, batch):
        result = super().__call__(batch)
        if not result:
            return {}
        valid = [b for b in batch if b is not None]
        result["translations"] = [b["translation"] for b in valid]
        result["sample_ids"]   = [b["sample_id"]   for b in valid]
        return result


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_v2g_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[VideoToGlossModel, dict[str, int], dict[int, str], int]:
    ckpt     = torch.load(checkpoint_path, map_location=device)
    vocab    = ckpt["vocab"]
    model    = VideoToGlossModel(num_classes=len(vocab))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    id2token = {v: k for k, v in vocab.items()}
    blank_id = vocab.get("<blank>", 0)
    print(
        f"[V2G] Loaded checkpoint  step={ckpt.get('global_step', '?')}  "
        f"best_wer={ckpt.get('best_metric', '?')}"
    )
    return model, vocab, id2token, blank_id


def load_g2t_model(
    model_dir: str,
    device: torch.device,
) -> tuple[M2M100ForConditionalGeneration, M2M100Tokenizer, int, int]:
    tokenizer    = M2M100Tokenizer.from_pretrained(model_dir)
    model        = M2M100ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device).eval()
    lsc_token_id = tokenizer.convert_tokens_to_ids("__lsc__")
    tgt_lang_id  = tokenizer.get_lang_id("ca")
    print(
        f"[G2T] Loaded model from {model_dir}  "
        f"lsc_token_id={lsc_token_id}  tgt_lang_id={tgt_lang_id}"
    )
    return model, tokenizer, lsc_token_id, tgt_lang_id


def load_cometkiwi(device: torch.device):
    if not _COMET_AVAILABLE:
        return None
    print(f"[COMET-Kiwi] Loading {COMETKIWI_MODEL} …")
    try:
        # Import COMET's built-in download helper
        from comet import download_model
        
        # This automatically resolves and returns the direct path to the .ckpt file
        checkpoint_path = download_model(COMETKIWI_MODEL)
        
        comet_model = load_from_checkpoint(checkpoint_path)
        comet_model.to(device)
        print("[COMET-Kiwi] Model ready.")
        return comet_model
    except Exception as e:
        print(f"[COMET-Kiwi] Failed to load: {e}")
        print("[COMET-Kiwi] Skipping — run with --skip_cometkiwi to suppress this.")
        return None

# ---------------------------------------------------------------------------
# Stage 1 — Video → Glosses
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stage1(
    v2g_model:  VideoToGlossModel,
    data_loader: DataLoader,
    id2token:   dict[int, str],
    device:     torch.device,
    fp16:       bool,
    blank_id:   int = 0,
) -> tuple[list[str], list[str], list[str]]:
    pred_glosses: list[str] = []
    ref_glosses:  list[str] = []
    ref_texts:    list[str] = []

    for batch in data_loader:
        if not batch:
            continue

        videos   = batch["videos"].to(device)
        vid_lens = batch["video_lengths"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            ctc_logits, _, input_lengths = v2g_model(videos, vid_lens)

        hyps = batch_ctc_greedy_decode(ctc_logits, input_lengths, id2token, blank_id)

        pred_glosses.extend(hyps)
        ref_glosses.extend(batch["references"])
        ref_texts.extend(batch["translations"])

    print(f"[Stage 1] {len(pred_glosses)} samples decoded.")
    return pred_glosses, ref_glosses, ref_texts


# ---------------------------------------------------------------------------
# Stage 2 — Glosses → Text
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stage2(
    g2t_model:       M2M100ForConditionalGeneration,
    tokenizer:       M2M100Tokenizer,
    gloss_sequences: list[str],
    lsc_token_id:    int,
    tgt_lang_id:     int,
    device:          torch.device,
    batch_size:      int = 16,
    num_beams:       int = 5,
    fp16:            bool = True,
) -> list[str]:
    predictions: list[str] = []

    for start in range(0, len(gloss_sequences), batch_size):
        chunk  = gloss_sequences[start : start + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        # Stamp __lsc__ at BOS (same as training preprocessing)
        inputs["input_ids"][:, 0] = lsc_token_id
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.cuda.amp.autocast(enabled=fp16):
            generated_ids = g2t_model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=128,
                num_beams=num_beams,
            )

        predictions.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )

    print(f"[Stage 2] {len(predictions)} translations generated.")
    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    device:     torch.device,
) -> dict[str, float]:
    """
    BERTScore F1 with xlm-roberta-large.

    xlm-roberta-large is the correct backbone for Catalan:
      - covers ca natively in its multilingual pretraining
      - standard choice in low-resource Romance language NLP evaluation
      - lang='ca' is passed so BERTScore uses the Catalan IDF weights
        (falls back gracefully to 'others' if not in its lookup table)
    """
    if not _BERTSCORE_AVAILABLE:
        return {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}

    print("[BERTScore] Computing with xlm-roberta-large …")
    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=BERTSCORE_MODEL,
        lang="ca",
        device=str(device),
        verbose=False,
        batch_size=32,
    )
    return {
        "bertscore_p":  round(P.mean().item(),  4),
        "bertscore_r":  round(R.mean().item(),  4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def compute_cometkiwi(
    hypotheses:  list[str],
    references:  list[str],
    comet_model,
) -> dict[str, float]:
    """
    COMET-Kiwi (QE) score using wmt23-cometkiwi-da-xl.

    Protocol follows audio/speech-translation papers (SeamlessM4T, NLLB,
    mSLAM): since the source is non-text (video / audio), we pass only
    (hypothesis, reference) — no source string. This is the standard
    reference-based QE evaluation mode for speech translation.

    The model returns scores in [0, 1]; higher is better.
    """
    if comet_model is None:
        return {"cometkiwi_mean": None, "cometkiwi_system": None}

    print("[COMET-Kiwi] Scoring …")

    # COMET-Kiwi with no source: pass hypothesis as 'mt', reference as 'ref',
    data = [
        {"mt": hyp, "src": ref}
        for hyp, ref in zip(hypotheses, references)
    ]

    # comet 2.2.x API: model.predict returns a Prediction namedtuple
    # with .scores (per-segment) and .system_score (corpus-level mean)
    output = comet_model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)

    seg_mean    = sum(output.scores) / len(output.scores)
    system_score = output.system_score

    return {
        "cometkiwi_mean":   round(seg_mean,    4),
        "cometkiwi_system": round(system_score, 4),
    }


def compute_translation_metrics(
    hypotheses:      list[str],
    references:      list[str],
    sacrebleu_metric,
    chrf_metric,
    comet_model,
    device:          torch.device,
    prefix:          str = "",
) -> dict[str, float]:
    refs_bleu = [[r] for r in references]

    bleu_result = sacrebleu_metric.compute(predictions=hypotheses, references=refs_bleu)
    chrf_result = chrf_metric.compute(predictions=hypotheses,      references=refs_bleu)

    metrics: dict[str, float] = {
        f"{prefix}bleu": round(bleu_result["score"], 2),
        f"{prefix}chrf": round(chrf_result["score"], 2),
    }

    # BERTScore
    bs = compute_bertscore(hypotheses, references, device)
    metrics.update({f"{prefix}{k}": v for k, v in bs.items()})

    # COMET-Kiwi
    ck = compute_cometkiwi(hypotheses, references, comet_model)
    metrics.update({f"{prefix}{k}": v for k, v in ck.items()})

    return metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(val, pct: bool = False) -> str:
    """Format a metric value, handling None gracefully."""
    if val is None:
        return "N/A (package not installed)"
    if pct:
        return f"{val * 100:.2f}%"
    return f"{val:.4f}"


def print_report(
    gloss_wer:       float,
    cascade_metrics: dict,
    oracle_metrics:  dict,
    n_samples:       int,
    v2g_checkpoint:  str,
    g2t_model_dir:   str,
    sample_tuples:   list[tuple],
    n_show:          int = 5,
) -> None:
    bar  = "=" * 70
    dash = "-" * 70

    print(f"\n{bar}")
    print("  CASCADE EVALUATION REPORT")
    print(f"  V2G  : {os.path.basename(v2g_checkpoint)}")
    print(f"  G2T  : {os.path.basename(g2t_model_dir.rstrip('/'))}")
    print(f"  N    : {n_samples} samples")
    print(bar)

    print(f"  Gloss WER  (stage 1 error)              : {gloss_wer * 100:.1f}%")
    print(dash)

    def _block(label: str, prefix: str, m: dict) -> None:
        print(f"  {label}")
        print(f"    BLEU            : {_fmt(m.get(f'{prefix}bleu'))}")
        print(f"    chrF            : {_fmt(m.get(f'{prefix}chrf'))}")
        print(f"    BERTScore F1    : {_fmt(m.get(f'{prefix}bertscore_f1'))}")
        print(f"    BERTScore P     : {_fmt(m.get(f'{prefix}bertscore_p'))}")
        print(f"    BERTScore R     : {_fmt(m.get(f'{prefix}bertscore_r'))}")
        print(f"    COMET-Kiwi mean : {_fmt(m.get(f'{prefix}cometkiwi_mean'))}")
        print(f"    COMET-Kiwi sys  : {_fmt(m.get(f'{prefix}cometkiwi_system'))}")

    _block("CASCADE   video → pred_gloss → text", "cascade_", cascade_metrics)
    print(dash)
    _block("ORACLE    ref_gloss → text  [stage-1 ceiling]", "oracle_", oracle_metrics)
    print(bar)

    def _gap(key: str) -> str:
        c = cascade_metrics.get(f"cascade_{key}")
        o = oracle_metrics.get(f"oracle_{key}")
        if c is None or o is None:
            return "N/A"
        return f"{o - c:+.4f}"

    print("  Gaps (oracle − cascade) — how much stage-1 errors cost:")
    print(f"    BLEU            : {_gap('bleu')}")
    print(f"    chrF            : {_gap('chrf')}")
    print(f"    BERTScore F1    : {_gap('bertscore_f1')}")
    print(f"    COMET-Kiwi mean : {_gap('cometkiwi_mean')}")
    print(bar)

    print(f"\n  SAMPLE PREDICTIONS  (first {n_show})")
    print(dash)
    for i, (ref_g, pred_g, ref_t, cas_t, orc_t) in enumerate(sample_tuples[:n_show]):
        print(f"  [{i + 1}]")
        print(f"    ref  gloss    : {ref_g}")
        print(f"    pred gloss    : {pred_g}")
        print(f"    ref  text     : {ref_t}")
        print(f"    cascade  text : {cas_t}")
        print(f"    oracle   text : {orc_t}")
        print()
    print(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cascade evaluation: Video → Gloss (CTC) → Text (M2M100)"
    )
    p.add_argument("--v2g_checkpoint",     required=True)
    p.add_argument("--g2t_model_dir",      required=True)
    p.add_argument("--data_file",          required=True)
    p.add_argument("--translation_field",  default="translation")
    p.add_argument("--vocab_path",         default=None)
    p.add_argument("--batch_size_v2g",     type=int, default=1)
    p.add_argument("--batch_size_g2t",     type=int, default=16)
    p.add_argument("--num_beams",          type=int, default=5)
    p.add_argument("--skip_frames_stride", type=int, default=2)
    p.add_argument("--num_workers",        type=int, default=4)
    p.add_argument("--fp16",               action="store_true", default=True)
    p.add_argument("--no_fp16",            dest="fp16", action="store_false")
    p.add_argument("--save_predictions",   default=None,
                   help="Write per-sample predictions to this .json file")
    p.add_argument("--skip_cometkiwi",     action="store_true", default=False,
                   help="Skip COMET-Kiwi (saves ~5GB VRAM / download time)")
    p.add_argument("--skip_bertscore",     action="store_true", default=False,
                   help="Skip BERTScore")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Vocab path ---------------------------------------------------------
    if args.vocab_path:
        vocab_path = args.vocab_path
    elif os.path.exists(
        os.path.join(os.path.dirname(args.v2g_checkpoint), "vocab.json")
    ):
        vocab_path = os.path.join(os.path.dirname(args.v2g_checkpoint), "vocab.json")
    else:
        vocab_path = os.path.join(os.path.dirname(args.data_file), "vocab.json")

    # ---- Load models --------------------------------------------------------
    v2g_model, vocab, id2token, blank_id = load_v2g_model(args.v2g_checkpoint, device)
    g2t_model, tokenizer, lsc_token_id, tgt_lang_id = load_g2t_model(
        args.g2t_model_dir, device
    )

    # COMET-Kiwi is heavy (~5GB); allow skipping via flag
    comet_model = None
    if not args.skip_cometkiwi:
        comet_model = load_cometkiwi(device)

    # Override BERTScore availability if flag passed
    global _BERTSCORE_AVAILABLE
    if args.skip_bertscore:
        _BERTSCORE_AVAILABLE = False

    # ---- Dataset + DataLoader -----------------------------------------------
    dataset    = CascadeEvalDataset(
        args.data_file,
        vocab_path,
        skip_frames_stride=args.skip_frames_stride,
        translation_field=args.translation_field,
    )
    collate_fn = CascadeCollate(vocab)
    loader     = DataLoader(
        dataset,
        batch_size=args.batch_size_v2g,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"\nDataset: {len(dataset)} samples  →  {args.data_file}\n")

    # ---- Stage 1 ------------------------------------------------------------
    print("─" * 42)
    print("[Stage 1]  Video → Glosses")
    print("─" * 42)
    pred_glosses, ref_glosses, raw_ref_texts = run_stage1(
        v2g_model, loader, id2token, device, args.fp16, blank_id
    )

    # Normalise reference texts through the tokenizer round-trip so scores
    # match what Seq2SeqTrainer reports (it decodes labels before metric calc)
    tokenizer.tgt_lang = "ca"
    encoded_refs = tokenizer(
        text_target=raw_ref_texts, max_length=128, truncation=True
    )["input_ids"]
    ref_texts = tokenizer.batch_decode(encoded_refs, skip_special_tokens=True)

    wer_metric = hf_evaluate.load("wer")
    gloss_wer  = wer_metric.compute(predictions=pred_glosses, references=ref_glosses)
    print(f"Intermediate gloss WER: {gloss_wer * 100:.1f}%")

    # ---- Stage 2a: Cascade --------------------------------------------------
    print("\n" + "─" * 42)
    print("[Stage 2a]  Cascade: pred_glosses → text")
    print("─" * 42)
    cascade_texts = run_stage2(
        g2t_model, tokenizer, pred_glosses,
        lsc_token_id, tgt_lang_id, device,
        args.batch_size_g2t, args.num_beams, args.fp16,
    )

    # ---- Stage 2b: Oracle ---------------------------------------------------
    print("\n" + "─" * 42)
    print("[Stage 2b]  Oracle: ref_glosses → text")
    print("─" * 42)
    oracle_texts = run_stage2(
        g2t_model, tokenizer, ref_glosses,
        lsc_token_id, tgt_lang_id, device,
        args.batch_size_g2t, args.num_beams, args.fp16,
    )

    # ---- Metrics ------------------------------------------------------------
    print("\nLoading sacrebleu and chrF …")
    sacrebleu = hf_evaluate.load("sacrebleu")
    chrf      = hf_evaluate.load("chrf")

    print("\n[Metrics] CASCADE")
    cascade_metrics = compute_translation_metrics(
        hypotheses=cascade_texts,
        references=ref_texts,
        sacrebleu_metric=sacrebleu,
        chrf_metric=chrf,
        comet_model=comet_model,
        device=device,
        prefix="cascade_",
    )

    print("\n[Metrics] ORACLE")
    oracle_metrics = compute_translation_metrics(
        hypotheses=oracle_texts,
        references=ref_texts,
        sacrebleu_metric=sacrebleu,
        chrf_metric=chrf,
        comet_model=comet_model,
        device=device,
        prefix="oracle_",
    )

    # ---- Report -------------------------------------------------------------
    sample_tuples = list(
        zip(ref_glosses, pred_glosses, ref_texts, cascade_texts, oracle_texts)
    )
    print_report(
        gloss_wer=gloss_wer,
        cascade_metrics=cascade_metrics,
        oracle_metrics=oracle_metrics,
        n_samples=len(pred_glosses),
        v2g_checkpoint=args.v2g_checkpoint,
        g2t_model_dir=args.g2t_model_dir,
        sample_tuples=sample_tuples,
    )

    # ---- Save predictions ---------------------------------------------------
    if args.save_predictions:
        output_list = [
            {
                "ref_gloss":    rg,
                "pred_gloss":   pg,
                "ref_text":     rt,
                "cascade_text": ct,
                "oracle_text":  ot,
            }
            for rg, pg, rt, ct, ot in sample_tuples
        ]
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=4)
        print(f"\nPredictions saved → {args.save_predictions}")

    # ---- JSON summary -------------------------------------------------------
    summary = {
        "n_samples":  len(pred_glosses),
        "gloss_wer":  round(gloss_wer * 100, 2),
        **cascade_metrics,
        **oracle_metrics,
    }
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()