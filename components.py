# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CenterNorm(nn.Module):
    """Applies Center Normalization over specified dimensions.

    This normalization technique centers the data to have a mean of zero and
    then scales it. The formula is:
        y = sqrt(D / (D - 1)) * (x - E[x])

    Where D is the number of elements being normalized. This is similar to
    LayerNorm but does not normalize the variance to 1. An optional affine
    transformation (gamma and beta) can be applied.

    Args:
        normalized_shape (int or tuple): The shape of the dimensions to normalize.
        eps (float): Kept for API compatibility with LayerNorm, but not used.
        elementwise_affine (bool): If True, this module has learnable affine
            parameters (gamma and beta).
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # D: The total number of elements in the normalized dimensions
        self.dim_size = math.prod(self.normalized_shape)

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        # Pre-calculate the scaling factor sqrt(D / (D-1))
        if self.dim_size > 1:
            scale_value = math.sqrt(self.dim_size / (self.dim_size - 1))
        else:
            scale_value = 1.0
        self.register_buffer('centernorm_scale_factor', torch.tensor(scale_value))

        # Dimensions to reduce over for mean calculation (the last D dims)
        self.reduction_dims = tuple(range(-len(self.normalized_shape), 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=self.reduction_dims, keepdim=True)
        centered_x = x - mean

        if self.dim_size > 1:
            scaled_x = centered_x * self.centernorm_scale_factor
        else:
            scaled_x = centered_x

        if self.elementwise_affine:
            scaled_x = self.gamma * scaled_x + self.beta

        return scaled_x

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class HQSAAttention(nn.Module):
    """Histogram Quantized Scalar Attention (HQSA).

    This module implements an efficient attention mechanism where keys and values
    are first aggregated into a fixed number of "bins". Queries then attend to
    these aggregated bin representations instead of every key, making the
    attention complexity sub-quadratic with respect to sequence length.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        num_bins (int): Number of bins to aggregate keys/values into.
        attn_dropout_prob (float): Dropout probability for attention weights.
    """
    def __init__(self, d_model: int, num_heads: int, num_bins: int, attn_dropout_prob: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_bins = num_bins
        self.attn_dropout_prob = attn_dropout_prob

        # Learnable per-head inverse-temperature for scaling bin assignment logits
        self.log_tau = nn.Parameter(torch.zeros(self.num_heads))

        # --- Multi-Prototype Binning Setup ---
        self.d_lat = 8
        self.W_bin = nn.Linear(self.d_head, self.d_lat, bias=False)
        self.prototypes = nn.Parameter(
            torch.randn(self.num_heads, self.num_bins, self.d_lat) / math.sqrt(self.d_lat)
        )

        # Standard linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        return_interpretability: bool = False,
        update_bins: bool = True,
        cached_bins: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, N, D_h = self.num_heads, self.num_bins, self.d_head

        # 1. Projections
        Q_proj = self.W_q(x).view(B, S, H, D_h).permute(0, 2, 1, 3)
        K_proj = self.W_k(x).view(B, S, H, D_h).permute(0, 2, 1, 3)
        V_proj = self.W_v(x).view(B, S, H, D_h).permute(0, 2, 1, 3)

        # 2. Key-to-Bin Soft Assignment
        z = self.W_bin(K_proj)
        tau = torch.exp(self.log_tau).clamp_min(1e-3)[None, :, None, None]
        proto_t = self.prototypes.transpose(1, 2).unsqueeze(0)
        logits = torch.matmul(z, proto_t) / (tau + 1e-8)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        attn_probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        # 3. Aggregate Keys and Values into Bins
        if update_bins:
            attn_probs_t = attn_probs.transpose(2, 3)
            weighted_K = torch.matmul(attn_probs_t, K_proj)
            weighted_V = torch.matmul(attn_probs_t, V_proj)
            counts = attn_probs.sum(dim=2, keepdim=True)
            K_binned = weighted_K / (counts + 1e-6)
            V_binned = weighted_V / (counts + 1e-6)
        else:
            assert cached_bins is not None, "cached_bins must be provided if update_bins is False"
            K_binned, V_binned = cached_bins

        # 4. Attention (Queries attend to Binned Keys)
        with torch.amp.autocast(device_type=x.device.type, enabled=True, dtype=torch.float32):
            attn_scores_raw = torch.matmul(Q_proj.float(), K_binned.transpose(-2, -1).float())
        attn_scores = attn_scores_raw / math.sqrt(D_h)

        if attention_mask is not None:
            query_padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(query_padding_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. Context Vector & Output
        context = torch.matmul(attn_weights.to(V_binned.dtype), V_binned)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.W_o(context)

        if return_interpretability:
            bin_ids = torch.argmax(attn_probs, dim=-1)
            return output, bin_ids, attn_probs, attn_weights, K_binned, V_binned
        else:
            return output, None, None, None, K_binned, V_binned 