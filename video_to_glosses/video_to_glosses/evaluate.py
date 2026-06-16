#!/usr/bin/env python3
"""
evaluate.py
-----------
Run CTC evaluation metrics on a saved checkpoint against any JSON split
(validation or test).

Usage:
    python evaluate.py \
        --checkpoint /path/to/best_model.pt \
        --data_file  /path/to/val.json \
        --output_dir /path/to/gloss_experiment \
        [--batch_size 2] \
        [--fp16] \
        [--save_predictions predictions.jsonl] \
        [--skip_frames_stride 1]

Outputs to stdout and optionally writes a JSONL file with per-sample
reference / hypothesis pairs for qualitative inspection.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SignVideoDataset, SignLanguageCollate
from model import VideoToGlossModel
from metrics import (
    batch_ctc_greedy_decode,
    compute_ctc_metrics,
    format_metrics,
    word_error_rate,
    match_error_rate,
    word_information_lost,
    character_error_rate,
    token_accuracy,
    sequence_accuracy,
    batch_ctc_beam_decode,
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[VideoToGlossModel, dict, dict]:
    """
    Load a VideoToGlossModel from a .pt checkpoint file.

    Returns:
        model     — loaded and eval()-mode model on `device`
        vocab     — token → id dict
        args_dict — training args dict stored in the checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab: dict[str, int] = ckpt["vocab"]
    args_dict: dict       = ckpt.get("training_args", {})

    model = VideoToGlossModel(num_classes=len(vocab))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    step  = ckpt.get("global_step", "?")
    epoch = ckpt.get("epoch", "?")
    bm    = ckpt.get("best_metric", "?")
    print(f"  Checkpoint: step={step}, epoch={epoch}, best_metric={bm}")
    return model, vocab, args_dict


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VideoToGlossModel,
    data_loader: DataLoader,
    id2token: dict[int, str],
    device: torch.device,
    fp16: bool,
    blank_id: int = 0,
    save_predictions_path: str | None = None,
) -> dict[str, float]:
    """
    Run greedy CTC decoding over the full data_loader and compute all metrics.

    Returns a dict of metric_name → value.
    """
    all_refs:  list[str] = []
    all_hyps:  list[str] = []
    predictions_log: list[dict] = []

    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        if not batch:
            continue

        videos      = batch["videos"].to(device)
        vid_lens    = batch["video_lengths"].to(device)
        refs        = batch["references"]       # list[str]

        with torch.cuda.amp.autocast(enabled=fp16):
            ctc_logits, _, input_lengths = model(videos, vid_lens)

        # hyps = batch_ctc_greedy_decode(ctc_logits, input_lengths, id2token, blank_id)
        hyps = batch_ctc_beam_decode(
            ctc_logits, input_lengths, id2token,
            beam_width=10,
            blank_id=blank_id,
        )
        
        all_refs.extend(refs)
        all_hyps.extend(hyps)
        total_samples += len(refs)

        if save_predictions_path:
            for ref, hyp in zip(refs, hyps):
                predictions_log.append({"reference": ref, "hypothesis": hyp})

        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0:
            partial = compute_ctc_metrics(all_refs, all_hyps)
            print(
                f"  [{batch_idx+1} batches / {total_samples} samples] "
                f"partial WER={partial['wer']*100:.1f}%"
            )

    # --- Save predictions JSONL ---
    if save_predictions_path:
        with open(save_predictions_path, "w", encoding="utf-8") as f:
            for entry in predictions_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nPredictions saved → {save_predictions_path}")

    return compute_ctc_metrics(all_refs, all_hyps)


# ---------------------------------------------------------------------------
# Detailed breakdown helpers
# ---------------------------------------------------------------------------

def print_detailed_report(
    metrics: dict[str, float],
    split_name: str,
    checkpoint_path: str,
):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Evaluation Report")
    print(f"  Checkpoint : {os.path.basename(checkpoint_path)}")
    print(f"  Split      : {split_name}")
    print(bar)
    print(f"  WER       (Word Error Rate)         : {metrics['wer']*100:6.2f}%")
    print(f"  MER       (Match Error Rate)        : {metrics['mer']*100:6.2f}%")
    print(f"  WIL       (Word Information Lost)   : {metrics['wil']*100:6.2f}%")
    print(f"  CER       (Character Error Rate)    : {metrics['cer']*100:6.2f}%")
    print(f"  Tok. Acc. (Token Accuracy)          : {metrics['tok_acc']*100:6.2f}%")
    print(f"  Seq. Acc. (Sequence Accuracy)       : {metrics['seq_acc']*100:6.2f}%")
    print(bar)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CTC checkpoint on a JSON data split."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint file (best_model.pt or checkpoint_step_N.pt)",
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to the JSON data file (val.json or test.json)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory containing vocab.json. If omitted, inferred from checkpoint directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Eval batch size (default: 2)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 autocast during inference (default: True)",
    )
    parser.add_argument(
        "--no_fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16",
    )
    parser.add_argument(
        "--skip_frames_stride",
        type=int,
        default=1,
        help="Frame subsampling stride (must match training; default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--save_predictions",
        default=None,
        help="If set, save per-sample ref/hyp pairs to this JSONL file",
    )
    parser.add_argument(
        "--split_name",
        default=None,
        help="Human-readable split label for the report (auto-detected if omitted)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Infer vocab directory ---
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        sys.exit(f"ERROR: vocab.json not found at {vocab_path}. "
                 "Pass --output_dir pointing to the directory that contains it.")

    # --- Load model ---
    model, vocab, ckpt_args = load_model_from_checkpoint(args.checkpoint, device)
    id2token: dict[int, str] = {v: k for k, v in vocab.items()}
    blank_id = vocab.get("<blank>", 0)

    # Stride falls back to whatever was used at training time
    stride = args.skip_frames_stride or ckpt_args.get("skip_frames_stride", 1)

    # --- Dataset ---
    dataset = SignVideoDataset(
        args.data_file,
        vocab_path,
        skip_frames_stride=stride,
    )
    collate_fn = SignLanguageCollate(vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Split name for report ---
    split_name = args.split_name or os.path.splitext(os.path.basename(args.data_file))[0]

    print(f"\nEvaluating {len(dataset)} samples from '{split_name}' split…")

    # --- Run ---
    metrics = evaluate(
        model,
        loader,
        id2token,
        device,
        fp16=args.fp16,
        blank_id=blank_id,
        save_predictions_path=args.save_predictions,
    )

    # --- Report ---
    print_detailed_report(metrics, split_name, args.checkpoint)

    # Also dump JSON for easy scripting
    metrics_out = {k: round(v * 100, 2) for k, v in metrics.items()}
    metrics_out["checkpoint"] = args.checkpoint
    metrics_out["split"]      = split_name
    print("JSON output:")
    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()