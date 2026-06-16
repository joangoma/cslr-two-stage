# train.py
import os
import json
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter

from dataclasses import dataclass
from dataset import SignVideoDataset, SignLanguageCollate
from model import VideoToGlossModel
from metrics import batch_ctc_greedy_decode, format_metrics, get_ctc_metric_collection


@dataclass
class CSLRTrainingArguments:
    input_dir: str    # contains vocab.json, train.json, val.json
    output_dir: str   

    # Eval / save / log cadence
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10

    # Optimisation
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    weight_decay: float = 0.01
    max_grad_norm: float = 5.0
    num_train_epochs: int = 80
    warmup_steps: int = 200          
    fp16: bool = True

    early_stopping_patience: int = 10
    metric_for_best_model: str = "wer"   
    greater_is_better: bool = False      

    # Data
    skip_frames_stride: int = 2
    dataloader_num_workers: int = 8

    # Loss mixing
    alpha: float = 1.0
    blank_penalty_lambda: float = 0.1   # tune between 0.05–0.3

    seed: int = 3435


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the step checkpoint with the highest step number, or None."""
    pattern = os.path.join(output_dir, "checkpoint_step_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None

    def _step(p):
        try:
            return int(os.path.basename(p)
                       .replace("checkpoint_step_", "")
                       .replace(".pt", ""))
        except ValueError:
            return -1

    return max(ckpts, key=_step)


def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    global_step, epoch, best_metric, vocab, args):
    torch.save({
        "global_step":          global_step,
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "best_metric":          best_metric,
        "vocab":                vocab,
        "training_args":        args.__dict__,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load all states in-place. Returns (global_step, start_epoch, best_metric)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    global_step  = ckpt["global_step"]
    epoch        = ckpt["epoch"]
    best_metric  = ckpt.get("best_metric", float("inf"))
    print(f"  ↳ Resumed from step {global_step} (epoch {epoch}), "
          f"best_metric={best_metric:.4f}")
    return global_step, epoch, best_metric


def prune_old_checkpoints(output_dir: str, keep: int = 2):
    """Delete all but the `keep` most recent step-checkpoints."""
    pattern = os.path.join(output_dir, "checkpoint_step_*.pt")
    ckpts = sorted(glob.glob(pattern), key=lambda p: int(
        os.path.basename(p).replace("checkpoint_step_", "").replace(".pt", "")
    ))
    for old in ckpts[:-keep]:
        os.remove(old)
        print(f"  🗑  Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, val_loader, ctc_loss_fn, ce_loss_fn,
                   id2token, device, fp16, alpha, val_metrics, blank_id=0):
    model.eval()
    val_metrics.reset()                      # clear state from any previous call

    val_ctc_total = val_aux_total = 0.0
    val_batches = 0

    for v_batch in val_loader:
        if not v_batch:
            continue
        val_batches += 1

        v_videos      = v_batch["videos"].to(device)
        v_vid_lens    = v_batch["video_lengths"].to(device)
        v_ctc_targets = v_batch["ctc_targets"]
        v_ctc_lens    = v_batch["ctc_lengths"]
        v_frame_tgts  = v_batch["frame_targets"].to(device)
        v_refs        = v_batch["references"]

        with torch.cuda.amp.autocast(enabled=fp16):
            v_ctc_logits, v_frame_logits, v_in_lens = model(v_videos, v_vid_lens)

            v_log_probs = F.log_softmax(v_ctc_logits, dim=-1)
            v_loss_ctc  = ctc_loss_fn(
                v_log_probs.float().cpu(),
                v_ctc_targets,
                v_in_lens.cpu(),
                v_ctc_lens,
            ).to(device)
            
            v_blank_log_probs = v_log_probs[:, :, blank_id]          # (T, B)
            v_blank_penalty   = v_blank_log_probs.exp().mean()        # mean blank prob across T and B
            v_loss_ctc        = v_loss_ctc + 0.1 * v_blank_penalty

            vT = v_frame_logits.size(1)
            v_down_tgts = v_frame_tgts[:, ::2][:, :vT]
            if v_down_tgts.size(1) < vT:
                v_down_tgts = F.pad(
                    v_down_tgts, (0, vT - v_down_tgts.size(1)), value=-1
                )
            v_loss_aux = ce_loss_fn(
                v_frame_logits.reshape(-1, v_frame_logits.size(-1)),
                v_down_tgts.reshape(-1),
            )

        val_ctc_total += v_loss_ctc.item()
        val_aux_total += v_loss_aux.item()

        hyps = batch_ctc_greedy_decode(v_ctc_logits, v_in_lens, id2token, blank_id)
        val_metrics.update(hyps, v_refs)     # accumulate across batches

    avg_ctc   = val_ctc_total / max(val_batches, 1)
    avg_aux   = val_aux_total / max(val_batches, 1)
    avg_total = avg_ctc + alpha * avg_aux

    computed = {k: v.item() for k, v in val_metrics.compute().items()}

    return {
        "val/loss":     avg_total,
        "val/ctc_loss": avg_ctc,
        "val/aux_loss": avg_aux,
        **{f"val/{k}": v for k, v in computed.items()},
        **computed,                          # flat keys for early-stopping lookup
    }


def log_model_summary(model, print_architecture=True):
    """Prints model architecture and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE & PARAMETER SUMMARY")
    print("="*60)
    
    # Optional: Print the full PyTorch network architecture structure
    # (Can be very long for DINOv2, so it's behind a flag)
    if print_architecture:
        print(model)
        print("-" * 60)
        
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    print(f"Frozen Parameters:     {frozen_params:,} ({(frozen_params/total_params)*100:.2f}%)")
    print("="*60 + "\n")
    
    return total_params, trainable_params, frozen_params

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = CSLRTrainingArguments(
        input_dir=os.environ.get("CSLR_INPUT_DIR", "/path/to/gloss_experiment/all_glosses"),
        output_dir=os.environ.get(
            "CSLR_OUTPUT_DIR",
            os.path.join(os.path.dirname(__file__), "..", "output"),
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Vocab ---
    vocab_path = os.path.join(args.input_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    id2token: dict[int, str] = {v: k for k, v in vocab.items()}
    blank_id = vocab.get("<blank>", 0)
    
    
    _train_data = json.load(open(os.path.join(args.input_dir, "train.json")))
    gloss_counts = Counter(
        g for item in _train_data
        for g in item["gloss_sequence"]   # adjust key to match your JSON structure
    )
    total_gloss_tokens = sum(gloss_counts.values())

    # Weight = log(total / count), capped at 10. Rare glosses get higher weight.
    ce_weights = torch.ones(len(vocab), device=device)
    for gloss, idx in vocab.items():
        count = gloss_counts.get(gloss, 1)
        ce_weights[idx] = min(
            math.log(total_gloss_tokens / count + 1), 10.0
        )
    ce_weights[blank_id] = 0.0   # never penalise blank in the aux CE loss

    # --- Datasets + Loaders ---
    collate_fn = SignLanguageCollate(vocab)
    
    train_dataset = SignVideoDataset(
        os.path.join(args.input_dir, "train.json"),
        vocab_path,
        skip_frames_stride=args.skip_frames_stride,
        augmentation_pipeline=None, 
    )
    val_dataset = SignVideoDataset(
        os.path.join(args.input_dir, "val.json"),
        vocab_path,
        skip_frames_stride=args.skip_frames_stride,
        augmentation_pipeline=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    # --- Model ---
    model = VideoToGlossModel(num_classes=len(vocab))
    model.to(device)
    
    metric_collection = get_ctc_metric_collection().to(device)

    total_p, train_p, frozen_p = log_model_summary(model, print_architecture=True)
    
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() 
                    if p.requires_grad and id(p) not in backbone_param_ids]

    decay_backbone, no_decay_backbone = [], []
    for name, param in model.backbone.named_parameters():
        if param.ndim == 1 or "bias" in name:
            no_decay_backbone.append(param)
        else:
            decay_backbone.append(param)

    decay_other, no_decay_other = [], []
    for name, param in model.named_parameters():
        if id(param) in backbone_param_ids or not param.requires_grad:
            continue
        if param.ndim == 1 or "bias" in name:
            no_decay_other.append(param)
        else:
            decay_other.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_backbone,    "lr": 5e-6,  "weight_decay": 0.01},
        {"params": no_decay_backbone, "lr": 5e-6,  "weight_decay": 0.0},
        {"params": decay_other,       "lr": 5e-4,  "weight_decay": 0.05},
        {"params": no_decay_other,    "lr": 5e-4,  "weight_decay": 0.0},
    ], betas=(0.9, 0.998))
    
    # --- Scheduler: linear warmup → cosine decay ---
    total_update_steps = (
        math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        * args.num_train_epochs
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(
            1, total_update_steps - args.warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- AMP scaler & losses ---
    scaler      = torch.cuda.amp.GradScaler(enabled=args.fp16)
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="mean")
    ce_loss_fn  = nn.CrossEntropyLoss(ignore_index=-1, weight=ce_weights)

    # -----------------------------------------------------------------------
    # Resume from checkpoint if available
    # -----------------------------------------------------------------------
    global_step      = 0
    start_epoch      = 1
    best_metric      = float("inf") if not args.greater_is_better else float("-inf")
    no_improve_count = 0

    latest_ckpt = find_latest_checkpoint(args.output_dir)
    if latest_ckpt:
        print(f"\n🔁 Resuming from checkpoint: {latest_ckpt}")
        global_step, start_epoch, best_metric = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler, device
        )
    else:
        print("\n🆕 No checkpoint found — starting from scratch.")

    print(f"\n{'='*60}")
    print(f"Total optimizer steps: {total_update_steps}")
    print(f"Early stopping on val/{args.metric_for_best_model} "
          f"(patience={args.early_stopping_patience})")
    print(f"{'='*60}\n")

    # Running accumulators for smooth train-loss logging
    accum_ctc = accum_aux = accum_n = 0.0

    for epoch in range(start_epoch, args.num_train_epochs + 1):
        model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if not batch:
                continue

            videos        = batch["videos"].to(device)
            video_lengths = batch["video_lengths"].to(device)
            ctc_targets   = batch["ctc_targets"]           # keep CPU for CTCLoss
            ctc_lengths   = batch["ctc_lengths"]           # keep CPU
            frame_targets = batch["frame_targets"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                ctc_logits, frame_logits, input_lengths = model(videos, video_lengths)

                log_probs = F.log_softmax(ctc_logits, dim=-1)
                loss_ctc  = ctc_loss_fn(
                    log_probs.float().cpu(),
                    ctc_targets,
                    input_lengths.cpu(),
                    ctc_lengths,
                ).to(device)
                
                blank_log_probs = log_probs[:, :, blank_id]          # (T, B)
                blank_penalty   = blank_log_probs.exp().mean()        # mean blank prob across T and B
                loss_ctc        = loss_ctc + args.blank_penalty_lambda * blank_penalty

                T_down          = frame_logits.size(1)
                down_frame_tgts = frame_targets[:, ::2][:, :T_down]
                if down_frame_tgts.size(1) < T_down:
                    down_frame_tgts = F.pad(
                        down_frame_tgts,
                        (0, T_down - down_frame_tgts.size(1)),
                        value=-1,
                    )
                loss_aux = ce_loss_fn(
                    frame_logits.reshape(-1, frame_logits.size(-1)),
                    down_frame_tgts.reshape(-1),
                )

                # Divide by accumulation steps ONLY for backward; log raw values
                loss_scaled = (loss_ctc) / args.gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Accumulate raw (unscaled) losses for logging
            accum_ctc += loss_ctc.item()
            accum_aux += loss_aux.item()
            accum_n   += 1

            # --- Optimizer step ---
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- Log training losses at optimizer-step granularity ---
                if global_step % args.logging_steps == 0:
                    avg_ctc_log   = accum_ctc / accum_n
                    avg_aux_log   = accum_aux / accum_n
                    avg_total_log = avg_ctc_log + args.alpha * avg_aux_log
                    current_lr    = scheduler.get_last_lr()[0]

                    print(
                        f"[E{epoch} S{global_step}] "
                        f"loss={avg_total_log:.4f}  ctc={avg_ctc_log:.4f}  "
                        f"aux={avg_aux_log:.4f}  lr={current_lr:.2e}"
                    )
                    accum_ctc = accum_aux = accum_n = 0.0

                # --- Validation + metrics ---
                if global_step % args.eval_steps == 0:
                    val_metrics = run_validation(
                        model, val_loader, ctc_loss_fn, ce_loss_fn,
                        id2token, device, args.fp16, args.alpha, metric_collection, blank_id,
                    )

                    primary = val_metrics[args.metric_for_best_model]

                    print(
                        f"\n--- Val @ step {global_step} ---\n"
                        f"  loss={val_metrics['val/loss']:.4f}  "
                        f"ctc={val_metrics['val/ctc_loss']:.4f}  "
                        f"aux={val_metrics['val/aux_loss']:.4f}\n"
                        f"{format_metrics({k: val_metrics[k] for k in ['wer', 'seq_acc']})}\n"
                    )

                    # --- Periodic step checkpoint ---
                    if global_step % args.save_steps == 0:
                        ckpt_path = os.path.join(
                            args.output_dir, f"checkpoint_step_{global_step}.pt"
                        )
                        save_checkpoint(
                            ckpt_path, model, optimizer, scheduler, scaler,
                            global_step, epoch, best_metric, vocab, args,
                        )
                        prune_old_checkpoints(args.output_dir, keep=2)
                        print(f"  💾 Checkpoint → {ckpt_path}")

                    # --- Best model + early stopping ---
                    improved = (
                        primary < best_metric if not args.greater_is_better
                        else primary > best_metric
                    )
                    if improved:
                        best_metric      = primary
                        no_improve_count = 0
                        best_path        = os.path.join(args.output_dir, "best_model.pt")
                        save_checkpoint(
                            best_path, model, optimizer, scheduler, scaler,
                            global_step, epoch, best_metric, vocab, args,
                        )
                        print(
                            f"  ✅ New best {args.metric_for_best_model}="
                            f"{best_metric:.4f} → {best_path}"
                        )
                    else:
                        no_improve_count += 1
                        print(
                            f"  ⏳ No improvement "
                            f"({no_improve_count}/{args.early_stopping_patience})"
                        )
                        if no_improve_count >= args.early_stopping_patience:
                            print("\n🛑 Early stopping triggered.")
                            return

                    model.train()


if __name__ == "__main__":
    main()