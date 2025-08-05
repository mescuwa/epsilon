# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
import torch
import torch.nn as nn
import torch.nn.utils as utils
from typing import Optional, Tuple

from components import HQSAAttention, CenterNorm


class FeedForward(nn.Module):
    """A standard Feed-Forward Network block with GELU and spectral normalization.

    Spectral normalization is applied to the linear layers to control their
    Lipschitz constant, which can help stabilize training of deep or
    recursive models.

    Args:
        d_model (int): Input and output dimension.
        ffn_dim (int): Inner dimension of the FFN.
        dropout_prob (float): Dropout probability.
    """
    def __init__(self, d_model: int, ffn_dim: int, dropout_prob: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.linear1 = utils.spectral_norm(self.linear1)
        self.linear2 = utils.spectral_norm(self.linear2)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EpsilonBlock(nn.Module):
    """A single recursive layer of the Epsilon Transformer.

    Structure: x -> Norm -> Attention -> Residual -> Norm -> FFN -> Residual.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads for HQSA.
        num_bins (int): Number of bins for HQSA.
        ffn_dim (int): Inner dimension of the FFN.
        dropout_prob (float): Dropout probability.
        alpha_res (float): Residual scaling factor.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_bins: int,
        ffn_dim: int,
        dropout_prob: float,
        alpha_res: float,
    ):
        super().__init__()
        self.alpha_res = alpha_res

        self.norm1 = CenterNorm(d_model)
        self.norm2 = CenterNorm(d_model)

        self.attention = HQSAAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_bins=num_bins,
            attn_dropout_prob=dropout_prob,
        )
        self.ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout_prob=dropout_prob)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        return_interpretability: bool = False,
        update_bins: bool = True,
        cached_bins: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Forward pass for the EpsilonBlock."""
        residual = x
        normed_x = self.norm1(x)
        attn_output, bin_ids, bin_dist, attn_wts, K_binned, V_binned = self.attention(
            normed_x,
            attention_mask=attention_mask,
            return_interpretability=return_interpretability,
            update_bins=update_bins,
            cached_bins=cached_bins,
        )
        x = residual + self.dropout1(attn_output * self.alpha_res)

        residual = x
        normed_x = self.norm2(x)
        ffn_output = self.ffn(normed_x)
        x = residual + self.dropout2(ffn_output * self.alpha_res)

        if return_interpretability:
            return x, bin_ids, bin_dist, attn_wts, K_binned, V_binned
        else:
            return x, None, None, None, K_binned, V_binned 