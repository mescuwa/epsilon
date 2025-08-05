# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
"""
Main Training Script for the Epsilon Transformer
=================================================
This script orchestrates the entire training and validation pipeline for the
Epsilon Transformer model on the IMDb dataset.

It includes support for modern, high-performance training techniques:
-   **Device Agnostic:** Runs on CUDA, MPS (Apple Silicon), or CPU.
-   **torch.compile:** JIT-compiles the model for significant speedups.
-   **Automatic Mixed Precision (AMP):** Uses float16 for faster training on
    supported GPUs.
-   **Advanced LR Scheduling:** Implements a linear warmup followed by a cosine
    annealing schedule.
-   **Gradient Clipping:** Prevents exploding gradients.
-   **Comprehensive Logging:** Uses `tqdm` for progress bars and integrates with
    `interpretability_utils` to save detailed diagnostics each epoch.
-   **Model Checkpointing:** Saves the model with the best validation accuracy.
"""
import argparse
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Project-specific imports
from data_utils import get_dataloaders
from model import EpsilonTransformer
from diagnostics import estimate_jacobian_spectral_norm
from interpretability_utils import (
    collect_halting_depths,
    flush_epoch_metrics,
    simple_ece,
    collect_ece,
    collect_bin_ids,
    collect_entropy_stats,
)


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seed(seed_value: int = 42) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Training / Validation routine
# -----------------------------------------------------------------------------

def train(args):
    """Main training loop encompassing setup, training, validation, and logging."""
    set_seed(42)

    # --- Device selection ---
    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data ---
    num_workers = (
        args.num_workers
        if args.num_workers >= 0
        else min(4, os.cpu_count() or 1)
    )
    train_loader, val_loader, vocab_size, pad_token_id = get_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        model_name=args.tokenizer_name,
        num_workers=num_workers,
    )
    print(f"Vocab size: {vocab_size} | Pad token: {pad_token_id}")

    # --- Model ---
    model = EpsilonTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_bins=args.num_bins,
        max_layers=args.max_layers,
        ffn_dim=args.ffn_dim,
        dropout_prob=args.dropout_prob,
        alpha_res=args.alpha_res,
        num_classes=args.num_classes,
        target_halting_mean=args.target_halting_mean,
        pad_token_id=pad_token_id,
        bin_update_frequency=args.bin_update_frequency,
    ).to(device)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Optional torch.compile for speed-ups
    if args.compile_mode != "default":
        print(f"Compiling model with mode='{args.compile_mode}' …")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print("Compilation succeeded.")
        except Exception as e:  # pragma: no cover – compile may fail on some setups
            print(f"Compilation failed ({e}); falling back to eager execution.")

    # AMP setup
    amp_enabled = args.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # Optimiser & LR scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.use_cosine_scheduler:
        total_steps = len(train_loader) * args.num_epochs
        warmup = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_steps
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - args.warmup_steps), eta_min=1e-7
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps]
        )

    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    # ---------------------------------------------------------------------
    # Epoch loop
    # ---------------------------------------------------------------------
    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        # --- Training ---
        model.train()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for step, batch in enumerate(train_iter):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                logits, step_probs, _, interp = model(
                    input_ids,
                    attention_mask,
                    return_interpretability=args.log_stats,
                )
                task_loss = criterion(logits, labels)
                kl_loss = (
                    model.calculate_kl_loss(step_probs, attention_mask)
                    if args.kl_loss_weight > 0
                    else 0.0
                )
                ent_loss = 0.0
                if (
                    args.ent_loss_weight > 0
                    and interp
                    and "bin_distribution" in interp
                ):
                    bin_probs = interp["bin_distribution"][
                        :, 0
                    ].float()  # first layer
                    active_mask = (
                        attention_mask.unsqueeze(1).unsqueeze(-1).bool()
                    )
                    ent = -(bin_probs * bin_probs.clamp_min(1e-9).log()).sum(-1)
                    masked = ent.masked_select(active_mask.squeeze(-1))
                    if masked.numel() > 0:
                        ent_loss = -masked.mean()
                total_loss = (
                    task_loss
                    + args.kl_loss_weight * kl_loss
                    + args.ent_loss_weight * ent_loss
                )

            scaler.scale(total_loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

            train_iter.set_postfix(
                {
                    "loss": f"{total_loss.item():.3f}",
                    "task": f"{task_loss.item():.3f}",
                    "kl": f"{kl_loss if isinstance(kl_loss, float) else kl_loss.item():.3f}",
                }
            )

        # --- Validation ---
        model.eval()
        val_loss, val_acc, num_batches = 0, 0, 0
        confidences, correctness = [], []
        pred_counter, label_counter = Counter(), Counter()
        sample_input_for_jac = None

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        with torch.no_grad():
            for step, batch in enumerate(val_iter):
                if args.limit_batches is not None and step >= args.limit_batches:
                    break
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)

                if step == 0 and args.estimate_jacobian:
                    embeds = model.embedding(ids) * math.sqrt(model.d_model)
                    sample_input_for_jac = model.pos_encoder(embeds).detach()

                with autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                    logits, _, halt_steps, interp = model(
                        ids, mask, return_interpretability=args.log_stats
                    )
                    loss = criterion(logits, lbl)

                probs = F.softmax(logits, dim=1)
                conf, preds = probs.max(dim=1)
                corr = preds == lbl

                val_loss += loss.item()
                val_acc += corr.float().mean().item()
                num_batches += 1
                pred_counter.update(preds.cpu().tolist())
                label_counter.update(lbl.cpu().tolist())

                if args.log_stats:
                    collect_halting_depths(
                        model.get_non_padded_halting_depths(halt_steps, mask)
                    )
                    confidences.append(conf.cpu())
                    correctness.append(corr.cpu())
                    if args.log_bins and interp and "bin_ids" in interp:
                        collect_bin_ids(interp["bin_ids"][:, 0].cpu(), mask.cpu())
                    if interp and "bin_distribution" in interp:
                        first_layer = interp["bin_distribution"][:, 0]
                        H = first_layer.shape[1]
                        per_head_H = [
                            simple_entropy(first_layer[:, h], mask) for h in range(H)
                        ]
                        collect_entropy_stats({"entropy": torch.tensor(per_head_H)})

            avg_val_loss = val_loss / num_batches if num_batches else 0
            avg_val_acc = val_acc / num_batches if num_batches else 0

        # --- Epoch summary & logging ---
        dur = time.time() - epoch_start
        print(
            f"\nEpoch {epoch+1} | Time: {dur:.1f}s | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
        )
        if len(pred_counter) == 1 and num_batches:
            print("🚨  Model predicted only one class – check learning dynamics!")

        # ECE
        if args.log_stats and confidences:
            epoch_ece = simple_ece(torch.cat(confidences), torch.cat(correctness))
            collect_ece(epoch_ece)
            print(f"  Epoch ECE: {epoch_ece:.4f}")

        # Jacobian norm estimate
        if sample_input_for_jac is not None:
            jac_norm = estimate_jacobian_spectral_norm(
                model.epsilon_block, sample_input_for_jac
            )
            print(f"  Epsilon Block Jacobian Norm (est.): {jac_norm:.4f}")

        # Flush interpretability data
        if args.log_stats:
            flush_epoch_metrics(Path(args.output_dir), epoch + 1, args.num_bins)

        # Checkpointing
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                cp_path = Path(args.output_dir) / "best_model.pt"
                to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(to_save.state_dict(), cp_path)
                print(f"  🎉 New best model saved to {cp_path} (Val Acc: {best_val_accuracy:.4f})")

        # KL annealing
        if args.kl_loss_weight > 0:
            progress = (epoch + 1) / args.num_epochs
            cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
            model._kl_scale = 0.1 + 0.9 * cosine_val  # 1.0 → 0.1
            print(f"  Annealed KL scale to {model._kl_scale:.3f}")

    print("\n--- Training complete ---")


# -----------------------------------------------------------------------------
# Small helper for entropy in validation (avoids circular import)
# -----------------------------------------------------------------------------

def simple_entropy(attn_probs: torch.Tensor, mask: torch.Tensor) -> float:
    log_p = attn_probs.clamp_min(1e-9).log()
    ent = -(attn_probs * log_p).sum(-1)
    valid = ent.masked_select(mask.bool())
    return valid.mean().item() if valid.numel() else 0.0


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Epsilon Transformer on IMDb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_model = p.add_argument_group("Model Hyperparameters")
    g_model.add_argument("--d_model", type=int, default=256)
    g_model.add_argument("--num_heads", type=int, default=4)
    g_model.add_argument("--num_bins", type=int, default=32)
    g_model.add_argument("--max_layers", type=int, default=12)
    g_model.add_argument("--ffn_dim", type=int, default=1024)
    g_model.add_argument("--dropout_prob", type=float, default=0.1)
    g_model.add_argument("--alpha_res", type=float, default=0.1)
    g_model.add_argument("--target_halting_mean", type=float, default=6.0)
    g_model.add_argument("--bin_update_frequency", type=int, default=1)

    g_train = p.add_argument_group("Training Hyperparameters")
    g_train.add_argument("--learning_rate", type=float, default=1e-4)
    g_train.add_argument("--weight_decay", type=float, default=0.01)
    g_train.add_argument("--batch_size", type=int, default=64)
    g_train.add_argument("--num_epochs", type=int, default=15)
    g_train.add_argument("--kl_loss_weight", type=float, default=0.01)
    g_train.add_argument("--ent_loss_weight", type=float, default=0.01)
    g_train.add_argument("--clip_grad_norm", type=float, default=1.0)

    g_data = p.add_argument_group("Data & Paths")
    g_data.add_argument("--tokenizer_name", default="bert-base-uncased")
    g_data.add_argument("--max_seq_length", type=int, default=512)
    g_data.add_argument("--output_dir", default="./epsilon_model_output")
    g_data.add_argument("--num_workers", type=int, default=-1)
    g_data.add_argument("--num_classes", type=int, default=2)
    g_data.add_argument("--limit_batches", type=int, default=None)

    g_perf = p.add_argument_group("Performance & Diagnostics")
    g_perf.add_argument("--force_cpu", action="store_true")
    g_perf.add_argument("--use_amp", action="store_true")
    g_perf.add_argument(
        "--compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    g_perf.add_argument("--use_cosine_scheduler", action="store_true")
    g_perf.add_argument("--warmup_steps", type=int, default=500)
    g_perf.add_argument("--log_stats", action="store_true")
    g_perf.add_argument("--log_bins", action="store_true")
    g_perf.add_argument("--estimate_jacobian", action="store_true")

    args = p.parse_args()
    train(args) 