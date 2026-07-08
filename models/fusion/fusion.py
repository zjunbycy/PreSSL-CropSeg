"""Temporal Fusion Modules for three strategies:

方案一 (Bottleneck / Mid-level): 1x1 Conv + Softmax at encoder-decoder junction
方案三 (Decision): per-frame logit averaging at output level
"""

import torch
import torch.nn as nn
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 方案一: Bottleneck Fusion (Mid-level)
# 在 Encoder 输出后、DPT 输入前，一次性坍缩 T 维度
# ──────────────────────────────────────────────────────────────────────

class BottleneckFusion(nn.Module):
    """Per-scale temporal attention: 1x1 Conv → Softmax → weighted sum.

    Args:
        in_channels: list of channel dims per scale [C1,C2,C3,C4]
        use_date_encoding: inject sinusoidal day-of-year encoding
        date_encoding_dim: dimension of date encoding
    """

    def __init__(
        self,
        in_channels: list,
        use_date_encoding: bool = True,
        date_encoding_dim: int = 6,
    ):
        super().__init__()
        self.use_date_encoding = use_date_encoding
        self.n_scales = len(in_channels)

        # Per-scale 1x1 Conv to produce scalar attention weights
        self.attention_convs = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1) for c in in_channels
        ])

        if use_date_encoding:
            self.date_fc = nn.Sequential(
                nn.Linear(date_encoding_dim, date_encoding_dim),
                nn.ReLU(),
                nn.Linear(date_encoding_dim, self.n_scales),
            )

    def _get_date_encoding(self, dates: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sinusoidal encoding of day-of-year values.

        Args:
            dates: (B, T) day-of-year integers (or None)

        Returns:
            (B, T, date_encoding_dim)
        """
        if dates is None:
            return None
        B, T = dates.shape
        # Normalize to [0, 2π]
        doy = dates.float() / 365.0 * 2 * torch.pi  # (B, T)
        enc = torch.stack([
            torch.sin(doy), torch.cos(doy),
            torch.sin(2 * doy), torch.cos(2 * doy),
            torch.sin(4 * doy), torch.cos(4 * doy),
        ], dim=-1).to(device)  # (B, T, 6) — keep it compact
        return enc

    def forward(
        self,
        features: list,          # [F1..F4], each (B, T, C_i, H_i, W_i)
        dates: Optional[torch.Tensor] = None,  # (B, T)
    ) -> list:
        """
        Returns:
            [S1..S4], each (B, C_i, H_i, W_i) — T collapsed
        """
        fused = []
        for i, feat in enumerate(features):
            B, T, C, H, W = feat.shape

            # Spatial attention weights
            attn = torch.stack([
                self.attention_convs[i](feat[:, t]) for t in range(T)
            ], dim=1)  # (B, T, 1, H, W)

            # Inject date bias
            if self.use_date_encoding and dates is not None:
                date_enc = self._get_date_encoding(dates, feat.device)
                bias = self.date_fc(date_enc)  # (B, T, n_scales)
                bias_i = bias[:, :, i:i + 1]  # (B, T, 1)
                attn = attn + bias_i.unsqueeze(-1).unsqueeze(-1)

            # Softmax over T
            attn = attn.flatten(3).mean(-1, keepdim=True).unsqueeze(-1)  # (B, T, 1, 1, 1)
            attn = torch.softmax(attn / (C ** 0.5), dim=1)

            # Weighted sum → collapse T
            out = (feat * attn).sum(dim=1)  # (B, C, H, W)

            fused.append(out)

        return fused


# ──────────────────────────────────────────────────────────────────────
# 方案三: Decision Fusion
# Per-frame DPT inference → soft voting at output level
# ──────────────────────────────────────────────────────────────────────

class DecisionFusion(nn.Module):
    """Soft voting over per-frame predictions.

    Each frame runs through decoder independently.
    Logits averaged → final prediction.
    No temporal interaction whatsoever.
    """

    def __init__(self):
        super().__init__()
        # No learnable params — pure averaging

    def forward(self, per_frame_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_frame_logits: (B, T, num_classes, H, W)

        Returns:
            (B, num_classes, H, W) — averaged logits
        """
        return per_frame_logits.mean(dim=1)
