# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

from epsilon_block import EpsilonBlock


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as described in "Attention Is All You Need"."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EpsilonTransformer(nn.Module):
    """Transformer with adaptive computation halting via recursive EpsilonBlock."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_bins: int,
        max_layers: int,
        ffn_dim: int,
        dropout_prob: float,
        alpha_res: float,
        num_classes: int,
        target_halting_mean: float,
        pad_token_id: int = 0,
        bin_update_frequency: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_layers = max_layers
        self.target_halting_mean = target_halting_mean
        self.pad_token_id = pad_token_id
        self.bin_update_frequency = bin_update_frequency

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout_prob, max_len=8192)
        self.epsilon_block = EpsilonBlock(
            d_model=d_model,
            num_heads=num_heads,
            num_bins=num_bins,
            ffn_dim=ffn_dim,
            dropout_prob=dropout_prob,
            alpha_res=alpha_res,
        )
        self.halting_fcs = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(max_layers)])
        self.classification_head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_token_id].fill_(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_interpretability: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, S = input_ids.shape
        device = input_ids.device

        h = self.embedding(input_ids) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        accumulated_output = torch.zeros_like(h)
        step_probabilities = torch.zeros(B, S, self.max_layers, device=device, dtype=h.dtype)
        accumulated_prob = torch.zeros(B, S, device=device, dtype=h.dtype)
        active_mask = attention_mask.bool()
        halting_step_count = torch.full((B, S), self.max_layers, dtype=torch.long, device=device)

        K_binned, V_binned = None, None
        interpretability_collectors = {"bin_ids": [], "bin_dist": [], "attn_wts": []}

        for t in range(self.max_layers):
            if not active_mask.any():
                break
            should_update_bins = (t % self.bin_update_frequency == 0) or (K_binned is None)
            h, b_ids, b_dist, a_wts, K_binned, V_binned = self.epsilon_block(
                h,
                return_interpretability=return_interpretability,
                update_bins=should_update_bins,
                cached_bins=(K_binned, V_binned),
            )

            halting_scores = self.halting_fcs[t](h).squeeze(-1)
            p_token_halt = torch.sigmoid(halting_scores)
            remainder = 1.0 - accumulated_prob
            p_step = torch.min(p_token_halt, remainder) * active_mask.to(h.dtype)
            step_probabilities[:, :, t] = p_step
            accumulated_prob += p_step
            accumulated_output += p_step.unsqueeze(-1) * h

            is_halted = (accumulated_prob >= 1.0 - 1e-6) & active_mask
            halting_step_count[is_halted] = torch.min(halting_step_count[is_halted], t + 1)
            active_mask = active_mask & ~is_halted

            if return_interpretability and all(x is not None for x in [b_ids, b_dist, a_wts]):
                interpretability_collectors["bin_ids"].append(b_ids)
                interpretability_collectors["bin_dist"].append(b_dist)
                interpretability_collectors["attn_wts"].append(a_wts)

        remaining_prob = (1.0 - accumulated_prob).unsqueeze(-1)
        final_output = accumulated_output + remaining_prob * h

        pooled_output = final_output[:, 0, :]
        logits = self.classification_head(pooled_output)

        interpretability_data = None
        if return_interpretability and interpretability_collectors["bin_ids"]:
            interpretability_data = {
                "bin_ids": torch.stack(interpretability_collectors["bin_ids"], dim=1),
                "bin_distribution": torch.stack(interpretability_collectors["bin_dist"], dim=1),
                "attn_weights": torch.stack(interpretability_collectors["attn_wts"], dim=1),
            }

        return logits, step_probabilities, halting_step_count, interpretability_data

    def calculate_kl_loss(self, step_probabilities: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        kl_scale = getattr(self, "_kl_scale", 1.0)
        active_mask = attention_mask.bool()
        if not active_mask.any():
            return torch.tensor(0.0, device=step_probabilities.device)

        p_dist = torch.cat(
            [
                step_probabilities,
                (1.0 - step_probabilities.sum(dim=-1, keepdim=True)).clamp_min(0.0),
            ],
            dim=-1,
        )
        p_geom = 1.0 / (self.target_halting_mean + 1e-8)
        q_dist = torch.zeros(self.max_layers + 1, device=p_dist.device, dtype=torch.float32)
        rem_prob = 1.0
        for t in range(self.max_layers):
            q_dist[t] = rem_prob * p_geom
            rem_prob *= 1.0 - p_geom
        q_dist[self.max_layers] = rem_prob
        q_dist /= q_dist.sum()

        kl_div = (p_dist * (p_dist.clamp_min(1e-9).log() - q_dist.log())).sum(dim=-1)
        masked_kl_div = kl_div.masked_fill(~active_mask, 0.0)
        mean_kl_div = masked_kl_div.sum() / active_mask.sum()
        return kl_scale * mean_kl_div

    def get_average_halting_depth(
        self, halting_step_count: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        active_mask = attention_mask.bool()
        if not active_mask.any():
            return None
        depths = (halting_step_count - 1).clamp_min(0).float()
        return depths[active_mask].mean()

    def get_non_padded_halting_depths(
        self, halting_step_count: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        active_mask = attention_mask.bool()
        if not active_mask.any():
            return None
        depths = (halting_step_count - 1).clamp_min(0).float()
        return depths[active_mask] 