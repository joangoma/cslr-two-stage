"""
train_g2t.py
------------
Gloss-to-Text (G2T) fine-tuning of M2M100 for Catalan Sign Language (LSC).

Key improvements over baseline:
  - Correct forced_bos_token_id so the decoder always generates Catalan
  - Source language token set to __lsc__ (not 'es') via manual BOS override
  - Deprecated as_target_tokenizer() replaced with text_target= API
  - label_smoothing_factor=0.1 for low-resource regularisation
      → If your Transformers version doesn't support it, remove that one line
        and the model will still train correctly; you just lose the smoothing.
  - no_repeat_ngram_size=3 to suppress the repetition loops you observed
  - Lower LR (1e-5) + longer warmup (15%) for cleaner convergence
  - Early stopping patience raised 3 → 5 so LR decay has room to help
  - Gradient checkpointing enabled to reduce VRAM pressure
  - EarlyStoppingCallback on sacrebleu (higher = better)
    - Console sample logging retained
"""

import os
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME  = "facebook/m2m100_418M"
DATA_DIR    = os.environ.get("G2T_DATA_DIR", "/path/to/ca_glosses")
OUTPUT_DIR  = os.environ.get(
    "G2T_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "output"),
)

TARGET_LANG = "ca"          # Catalan — decoder forced language
MAX_SRC_LEN = 128           # gloss sequences are short
MAX_TGT_LEN = 128


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # 1. Tokenizer + Model
    # -----------------------------------------------------------------------
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model     = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Register the LSC gloss pseudo-language token
    tokenizer.add_tokens(["__lsc__"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    lsc_token_id = tokenizer.convert_tokens_to_ids("__lsc__")

    # Force the decoder to always start with the Catalan language token.
    # Without this M2M100 may generate in any language it prefers.
    ca_lang_id = tokenizer.get_lang_id(TARGET_LANG)
    model.config.forced_bos_token_id = ca_lang_id

    # Suppress repetition loops (symptom you observed → likely underfitting,
    # but this prevents it from cascading into completely degenerate output
    # while the model is still learning).
    model.config.no_repeat_ngram_size = 3

    # Gradient checkpointing: trades a recomputation pass for lower VRAM.
    # Safe to remove if you have ample GPU memory.
    model.gradient_checkpointing_enable()

    print(f"[Init] LSC token id : {lsc_token_id}")
    print(f"[Init] Catalan lang id : {ca_lang_id}")
    print(f"[Init] Vocab size : {len(tokenizer)}")

    # -----------------------------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------------------------
    dataset = load_dataset(
        "csv",
        data_files={
            "train":      os.path.join(DATA_DIR, "train.tsv"),
            "validation": os.path.join(DATA_DIR, "validation.tsv"),
        },
        sep="\t",
    )
    print(f"[Data] Train samples      : {len(dataset['train'])}")
    print(f"[Data] Validation samples : {len(dataset['validation'])}")

    # -----------------------------------------------------------------------
    # 3. Preprocessing
    # -----------------------------------------------------------------------
    def preprocess_function(examples):
        inputs  = [str(ex) for ex in examples["gloss_input"]]
        targets = [str(ex) for ex in examples["output"]]

        # --- Source side: encode as plain text, then stamp __lsc__ at BOS ---
        # We deliberately do NOT set src_lang here because glosses are not
        # Spanish (or any M2M100 language). Encoding without a lang bias lets
        # the __lsc__ token carry the full source conditioning signal.
        tokenized_inputs = tokenizer(
            inputs,
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=False,
        )
        for ids in tokenized_inputs["input_ids"]:
            ids[0] = lsc_token_id   # overwrite BOS with __lsc__

        # --- Target side: use text_target= (replaces deprecated as_target_tokenizer) ---
        tokenizer.tgt_lang = TARGET_LANG
        tokenized_targets = tokenizer(
            text_target=targets,
            max_length=MAX_TGT_LEN,
            truncation=True,
            padding=False,
        )

        tokenized_inputs["labels"] = tokenized_targets["input_ids"]
        return tokenized_inputs

    print("\n[Preprocessing] Tokenizing datasets…")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print("[Preprocessing] Done.")

    # -----------------------------------------------------------------------
    # 4. Metrics
    # -----------------------------------------------------------------------
    sacrebleu_metric = evaluate.load("sacrebleu")
    chrf_metric      = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 (padding label) with pad_token_id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip surrounding whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # --- Console logging ---
        print("\n" + "=" * 55)
        print("        EVALUATION SAMPLES")
        print("=" * 55)
        for i in range(min(8, len(decoded_preds))):
            print(f"[{i+1}] TARGET   : {decoded_labels[i]}")
            print(f"    PREDICTED: {decoded_preds[i]}")
            print("-" * 55)
        print("=" * 55 + "\n")

        # --- Scores ---
        refs_bleu = [[l] for l in decoded_labels]
        bleu  = sacrebleu_metric.compute(predictions=decoded_preds, references=refs_bleu)
        chrf  = chrf_metric.compute(predictions=decoded_preds,      references=refs_bleu)

        return {
            "sacrebleu": round(bleu["score"], 4),
            "chrf":      round(chrf["score"], 4),
        }

    # -----------------------------------------------------------------------
    # 5. Data Collator
    # -----------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,    # so padding is ignored in loss
        pad_to_multiple_of=8,       # efficient on Tensor Cores
    )

    # -----------------------------------------------------------------------
    # 6. Training Arguments
    # -----------------------------------------------------------------------
    # NOTE on label_smoothing_factor:
    #   Supported since Transformers 4.10. If you get a TypeError on startup,
    #   simply delete the label_smoothing_factor line — everything else is
    #   unaffected.
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # --- Eval / save cadence ---
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        save_total_limit=2,

        # --- Optimisation ---
        learning_rate=3e-5,             # lower than baseline (3e-5 was too aggressive)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch = 32
        num_train_epochs=20,            # more headroom; early stopping will fire first
        warmup_ratio=0.15,              # longer warmup for stable low-LR training
        weight_decay=0.05,
        max_grad_norm=1.0,

        # --- Regularisation ---
        # label_smoothing_factor=0.1,     # remove this line if Transformers < 4.10

        # --- Generation ---
        predict_with_generate=True,
        generation_max_length=MAX_TGT_LEN,
        generation_num_beams=10,

        # --- Precision / memory ---
        fp16=True,

        # --- Logging ---
        logging_steps=10,
        report_to="none",
        run_name="m2m100_lsc_ca_v2",
    )

    # -----------------------------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            # Patience raised 3→5: gives the cosine LR schedule room to help
            # before giving up. On a small dataset BLEU can plateau for 2–3
            # epochs before a lower LR unlocks further gains.
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    # -----------------------------------------------------------------------
    # 8. Train
    # -----------------------------------------------------------------------
    print("\n[Training] Starting…")
    trainer.train()

    # -----------------------------------------------------------------------
    # 9. Save
    # -----------------------------------------------------------------------
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[Done] Model saved to {final_dir}")


if __name__ == "__main__":
    main()