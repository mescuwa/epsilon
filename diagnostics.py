# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
import torch
import torch.nn as nn
from torch.autograd.functional import jvp, vjp
from typing import Optional


def estimate_jacobian_spectral_norm(block: nn.Module, input_tensor: torch.Tensor,
                                    num_iterations: int = 10, tol: float = 1e-3) -> float:
    """Estimates the spectral norm (max singular value) of a block's Jacobian.

    The spectral norm of a Jacobian J is its largest singular value, which
    governs the maximum stretching factor of the function. It is a key
    indicator of model stability and sensitivity to input perturbations.
    A high spectral norm can be associated with exploding gradients.

    This function uses the power iteration method on the matrix J^T @ J,
    which converges to the largest eigenvalue. The square root of this
    eigenvalue is the spectral norm of J. This is done efficiently without
    materializing the full Jacobian.

    Args:
        block (nn.Module): The PyTorch module whose Jacobian norm is needed.
        input_tensor (torch.Tensor): A sample input tensor for the block.
            Shape should match the block's expected input (e.g., B, S, D).
            Only the shape and device matter; values can be random.
        num_iterations (int): Number of power iterations. More iterations
            yield a more accurate estimate.
        tol (float): Tolerance for convergence (currently unused, as a fixed
            number of iterations are performed).

    Returns:
        float: The estimated spectral norm.
    """
    device = input_tensor.device
    # Use a single example from the batch for Jacobian estimation, as JVP/VJP
    # are often cleaner with functions on single data points.
    batch_size, seq_len, d_model = input_tensor.shape
    input_flat_dim = seq_len * d_model
    block.eval()  # Ensure dropout/etc are disabled for consistent Jacobian

    # Define a function that maps a flattened input to a flattened output.
    # This is the standard interface required by jvp and vjp.
    def _func_flat(x_flat: torch.Tensor) -> torch.Tensor:
        # Reshape to the block's expected input shape (1, S, D)
        x = x_flat.view(1, seq_len, d_model)
        # Handle blocks that return tuples (like EpsilonBlock)
        output_tensor_tuple = block(x)
        if isinstance(output_tensor_tuple, tuple):
            actual_output_tensor = output_tensor_tuple[0]
        else:
            actual_output_tensor = output_tensor_tuple
        return actual_output_tensor.view(-1)  # Flatten for JVP/VJP

    # Use the first element of the batch as the reference input for the Jacobian
    x_ref_flat = input_tensor[0].view(-1).detach()

    # Initialize a random vector 'v' for power iteration
    v = torch.randn(input_flat_dim, device=device)
    v = v / torch.norm(v)

    with torch.no_grad():
        for _ in range(num_iterations):
            # Power iteration step: v = (J^T @ J) @ v
            # 1. Apply Jacobian transpose: J^T @ v (using vector-Jacobian product)
            _, Jt_v = vjp(_func_flat, x_ref_flat, v)
            # 2. Apply Jacobian: J @ (J^T @ v) (using Jacobian-vector product)
            _, J_Jt_v = jvp(_func_flat, x_ref_flat, Jt_v)

            # Normalize the vector for the next iteration
            norm_J_Jt_v = torch.norm(J_Jt_v)
            if norm_J_Jt_v < 1e-6:  # Avoid division by zero
                # Re-initialize if vector collapses (rare)
                v = torch.randn(input_flat_dim, device=device)
                v = v / torch.norm(v)
                continue
            v = J_Jt_v / norm_J_Jt_v

    # After convergence, the spectral norm is approximated by ||J @ v||.
    # Alternatively, the largest eigenvalue of J^T@J is ||(J^T@J)@v||.
    # The spectral norm is the sqrt of that eigenvalue.
    with torch.no_grad():
        _, Jt_v = vjp(_func_flat, x_ref_flat, v)
        _, J_Jt_v = jvp(_func_flat, x_ref_flat, Jt_v)

    spectral_norm_squared = torch.norm(J_Jt_v)
    spectral_norm = torch.sqrt(spectral_norm_squared).item()

    return spectral_norm


def hard_bin_metrics(bin_ids: torch.Tensor, n_bins: int) -> tuple:
    """Calculates metrics for hard bin assignments.

    Args:
        bin_ids (torch.Tensor): 1-D tensor of integer bin assignments for
            a set of tokens (padding should already be removed).
        n_bins (int): The total number of bins available.

    Returns:
        A tuple containing:
        - float: Entropy of the bin distribution in bits (-sum(p*log2(p))).
        - int: The number of bins that were actively used (count > 0).
        - float: The percentage of tokens assigned to the most-used bin.
    """
    if bin_ids.numel() == 0:
        return 0.0, 0, 0.0

    counts = torch.bincount(bin_ids, minlength=n_bins).float()
    total = counts.sum()

    if total.item() == 0:
        return 0.0, 0, 0.0

    probs = counts / total
    # Calculate entropy only for non-zero probabilities to avoid log(0) -> NaN
    nz_probs = probs[probs > 0]
    entropy_bits = -(nz_probs * nz_probs.log2()).sum().item()

    active_bins = (counts > 0).sum().item()
    top_bin_percentage = (counts.max() / total).item() * 100.0

    return entropy_bits, active_bins, top_bin_percentage


def soft_entropy(attn_probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Calculates the average entropy of soft probability distributions.

    This is useful for measuring the "confidence" of the soft bin assignments
    in HQSA. A high entropy means assignments are diffuse; low entropy means
    assignments are sharp (confident).

    Args:
        attn_probs (torch.Tensor): Tensor of probability distributions, where the
            last dimension sums to 1. Shape: (..., N_distributions).
        mask (Optional[torch.Tensor]): A boolean mask to select which distributions
            to include in the average. Shape must be broadcastable to
            `attn_probs`'s shape without the last dimension.

    Returns:
        float: The average entropy in bits, averaged over all valid distributions.
    """
    # Add a small epsilon for numerical stability with log2
    log_p = attn_probs.clamp_min(1e-9).log2()
    # Per-distribution entropy: -sum(p * log2(p)) over the distribution dim
    token_entropies = -(attn_probs * log_p).sum(dim=-1)

    if mask is not None:
        mask = mask.bool()
        # Select entropies of active elements only
        valid_entropies = token_entropies.masked_select(mask)
        if valid_entropies.numel() > 0:
            avg_entropy = valid_entropies.mean().item()
        else:
            avg_entropy = 0.0  # No valid tokens/distributions to average
    else:
        # If no mask, average over all provided distributions
        avg_entropy = token_entropies.mean().item()

    return avg_entropy 