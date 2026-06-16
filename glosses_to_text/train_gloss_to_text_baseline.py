import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# --- CONFIGURATION ---
MODEL_NAME = "facebook/m2m100_418M"
DATA_DIR = os.environ.get("G2T_DATA_DIR", "/path/to/ca_glosses/min_freq")
OUTPUT_DIR = os.environ.get(
    "G2T_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "output_f5"),
)

def main():
    # 2. Load Tokenizer and Model
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    tokenizer.add_tokens(["__lsc__"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    lsc_token_id = tokenizer.convert_tokens_to_ids("__lsc__")
    
    # 3. Load Dataset
    dataset = load_dataset("csv", data_files={
        "train": os.path.join(DATA_DIR, "train.tsv"),
        "validation": os.path.join(DATA_DIR, "validation.tsv")
    }, sep="\t")

    # 4. Preprocessing Function
    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples["gloss_input"]]
        targets = [str(ex) for ex in examples["output"]]
        
        tokenizer.src_lang = "es" 
        tokenizer.tgt_lang = "ca" 
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        for input_ids in model_inputs["input_ids"]:
            input_ids[0] = lsc_token_id
            
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("--- Tokenizing Datasets ---")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # 5. Setup Metrics & Log Generation Pairs
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Decode both predictions and targets
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # --- LOGGING TO CONSOLE / SLURM LOGS ---
        # Print a small subset (e.g., first 5) so your text logs don't get massive
        print("\n" + "="*50)
        print(f"      EVALUATION SAMPLES FOR CURRENT EPOCH")
        print("="*50)
        for i in range(min(5, len(decoded_preds))):
            print(f"Sample {i+1}:")
            print(f"  [PREDICTED]: {decoded_preds[i]}")
            print(f"  [TARGET]   : {decoded_labels[i]}")
            print("-" * 30)
        print("="*50 + "\n")
        
        # Compute metric scores
        decoded_labels_bleu = [[label] for label in decoded_labels]
        bleu_score = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)
        chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels_bleu)

        return {
            "sacrebleu": bleu_score["score"],
            "chrf": chrf_score["score"]
        }

    # 6. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 7. Regularized and Optimized Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        save_total_limit=2,
        warmup_ratio=0.1,
        weight_decay=0.05,                  # Stiffer penalty to fight overfitting
        num_train_epochs=15,
        predict_with_generate=True,  
        generation_max_length=128,
        fp16=True,
        logging_steps=10,
        report_to="none",
        run_name="m2m100_lsc_to_ca_optimized",
        load_best_model_at_end=True,        # Automatically roll back to peak performance
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        generation_num_beams=5,
    )

    # 8. Initialize Trainer with Early Stopping Callback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # 9. Start Training
    print("--- Starting Training Session ---")
    trainer.train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"Model successfully saved to {OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    main()