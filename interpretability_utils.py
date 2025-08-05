# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
"""
Interpretability Utilities
==========================
A lightweight, sidecar module for recording, aggregating, and exporting
diagnostic and interpretability data from the Epsilon model during training.

Design Philosophy:
- **Minimal Overhead:** The collection functions are simple appends that move
  data to the CPU, minimizing their impact on the GPU training loop.
- **Decoupled Logic:** The main training script calls these utilities but doesn't
  contain any complex logging or file I/O logic itself.
- **Offline Analysis:** The module saves data in simple formats (NPZ, CSV, JSON)
  that are easy to load into a separate environment (like a Jupyter notebook)
  for plotting and detailed analysis.

Workflow:
1.  During each training/validation step, the main loop calls `collect_*`
    functions to store batch-level data (e.g., halting depths, bin IDs) in
    in-memory buffers.
2.  At the end of each epoch, the main loop calls `flush_epoch_metrics()`.
3.  This `flush` function aggregates the data from all batches, computes
    epoch-level statistics, saves them to disk, and clears the buffers for the
    next epoch.
"""
import numpy as np
import json
import pathlib
import torch
import csv
from typing import List, Dict, Tuple

from diagnostics import hard_bin_metrics

# --- In-memory Buffers to store data collected across batches within an epoch ---
_halting_depths_buffer: List[torch.Tensor] = []
_ece_values_buffer: List[float] = []
_entropy_batches_buffer: List[Dict[str, torch.Tensor]] = []
_bin_id_data_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []


# --- Collector Functions (called multiple times per epoch from the train loop) ---

def collect_halting_depths(depths_tensor: torch.Tensor) -> None:
    """Stores a tensor of halting depths from a single batch.

    Args:
        depths_tensor: A 1-D tensor of halting depths for non-padded tokens.
    """
    _halting_depths_buffer.append(depths_tensor.detach().cpu())


def collect_ece(value: float) -> None:
    """Stores the Expected Calibration Error (ECE) value from a batch."""
    _ece_values_buffer.append(value)


def collect_entropy_stats(stats: Dict[str, torch.Tensor]) -> None:
    """Stores a dictionary of entropy stats from a batch."""
    _entropy_batches_buffer.append(stats)


def collect_bin_ids(batch_bin_ids: torch.Tensor, batch_attention_mask: torch.Tensor) -> None:
    """Stores the hard bin assignments and attention mask from a batch.

    Args:
        batch_bin_ids (torch.Tensor): Tensor of bin IDs (B, H, S).
        batch_attention_mask (torch.Tensor): Attention mask (B, S).
    """
    _bin_id_data_buffer.append((batch_bin_ids, batch_attention_mask))


# --- Flush Function (called once per epoch) ---

def flush_epoch_metrics(output_path: pathlib.Path, epoch: int, num_bins_config: int) -> None:
    """Aggregates all collected data for the epoch, saves to disk, and clears buffers."""
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Flushing Epoch {epoch} Interpretability Metrics ---")

    # --- Halting Depths ---
    if _halting_depths_buffer:
        try:
            depth_arr = torch.cat(_halting_depths_buffer).numpy()
            save_file = output_path / f"halting_epoch{epoch}.npz"
            np.savez_compressed(save_file, depths=depth_arr)
            print(f"[Interpretability] Saved halting depths (shape {depth_arr.shape}) to {save_file}")
            if depth_arr.size > 0:
                print(f"[Interpretability] Epoch Avg Halting Depth: {depth_arr.mean():.2f}")
        except Exception as e:
            print(f"Error saving halting depths NPZ: {e}")
    _halting_depths_buffer.clear()

    # --- Bin Diagnostic Metrics ---
    if _bin_id_data_buffer:
        try:
            csv_path = output_path / f"bin_diagnostics_epoch{epoch}.csv"
            print(f"[Interpretability] Aggregating bin diagnostics for {len(_bin_id_data_buffer)} batches...")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "layer", "head", "entropy_bits", "active_bins", "top_bin_percentage"])

                num_heads = _bin_id_data_buffer[0][0].shape[1]
                for head_idx in range(num_heads):
                    all_head_ids_list = []
                    for batch_ids, batch_mask in _bin_id_data_buffer:
                        head_ids = batch_ids[:, head_idx, :]
                        valid_ids = head_ids[batch_mask.bool()]
                        if valid_ids.numel() > 0:
                            all_head_ids_list.append(valid_ids)
                    if all_head_ids_list:
                        all_head_ids_flat = torch.cat(all_head_ids_list)
                        entropy, active, top_pct = hard_bin_metrics(all_head_ids_flat, n_bins=num_bins_config)
                        writer.writerow([epoch, 0, head_idx, f"{entropy:.4f}", active, f"{top_pct:.2f}"])
                    else:
                        writer.writerow([epoch, 0, head_idx, 0.0, 0, 0.0])
            print(f"[Interpretability] Saved bin diagnostics to {csv_path}")
        except Exception as e:
            print(f"Error saving bin diagnostics CSV: {e}")
    _bin_id_data_buffer.clear()

    # --- Expected Calibration Error (ECE) ---
    if _ece_values_buffer:
        try:
            ece_mean = float(np.mean(_ece_values_buffer))
            save_file = output_path / f"ece_epoch{epoch}.json"
            with open(save_file, "w") as fp:
                json.dump({"ece_mean": ece_mean, "ece_batch_values": _ece_values_buffer}, fp, indent=2)
            print(f"[Interpretability] Saved ECE (mean: {ece_mean:.4f}) to {save_file}")
        except Exception as e:
            print(f"Error saving ECE JSON: {e}")
    _ece_values_buffer.clear()

    # --- Entropy Stats (from soft assignments) ---
    if _entropy_batches_buffer:
        try:
            ent_arr = torch.stack([s['entropy'] for s in _entropy_batches_buffer]).numpy()
            save_file = output_path / f"entropy_epoch{epoch}.npz"
            np.savez_compressed(save_file, entropy=ent_arr)
            print(f"[Interpretability] Saved soft entropy stats (shape {ent_arr.shape}) to {save_file}")
        except Exception as e:
            print(f"Error saving entropy NPZ: {e}")
    _entropy_batches_buffer.clear()


def simple_ece(confidences: torch.Tensor, correctness: torch.Tensor, bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) in a simple, vectorized way."""
    if confidences.numel() == 0:
        return 0.0
    confidences = confidences.cpu().numpy()
    correctness = correctness.cpu().numpy().astype(float)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    total_count = len(confidences)
    for i in range(bins):
        if i < bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        else:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count_in_bin = in_bin.sum()
        if count_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += abs(accuracy_in_bin - confidence_in_bin) * (count_in_bin / total_count)
    return float(ece) 