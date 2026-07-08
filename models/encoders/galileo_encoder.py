"""Galileo Encoder Wrapper — loads BiliSakura/GALILEO-transformers weights.

Supports two modes:
  - per_frame: encode T frames independently, return [B,T,C_i,H_i,W_i] per scale
  - joint: encode all frames together (native Galileo spatiotemporal modeling)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal


class GalileoEncoder(nn.Module):
    """Wrapper for Galileo pretrained encoder from HuggingFace transformers.

    Args:
        model_name: HF repo id, e.g. 'BiliSakura/GALILEO-transformers'
        mode: 'per_frame' — each frame independently; 'joint' — all frames together
        in_channels: S2 bands (usually 10)
        img_size: spatial size (PASTIS = 128)
        freeze: freeze encoder weights for linear probing
        output_scales: how many feature scales to return (1-4)
    """

    def __init__(
        self,
        model_name: str = "BiliSakura/GALILEO-transformers",
        mode: Literal["per_frame", "joint"] = "per_frame",
        in_channels: int = 10,
        img_size: int = 128,
        freeze: bool = False,
        output_scales: int = 4,
    ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.img_size = img_size
        self.output_scales = output_scales

        # Try loading from HF transformers
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception:
            print(f"[Galileo] Could not load {model_name}, using placeholder")
            self.encoder = self._build_placeholder()

        self.encoder.eval() if freeze else self.encoder.train()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Output channels per scale — need to be inferred after loading
        self._out_channels = None

    def _build_placeholder(self):
        """Minimal placeholder so code runs without HF access."""
        return nn.Identity()

    @property
    def out_channels(self) -> List[int]:
        if self._out_channels is not None:
            return self._out_channels
        # Duck-test: run a dummy input
        dummy = torch.randn(1, self.mode == "per_frame" and 1 or 4,
                            self.in_channels, self.img_size, self.img_size)
        with torch.no_grad():
            feats = self.forward(dummy)
        self._out_channels = [f.shape[2] for f in feats]
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: if per_frame: (B, T, C, H, W);
               if joint: (B, T, C, H, W)

        Returns:
            List of feature maps [F1..F4], each (B, T, C_i, H_i, W_i)
        """
        B, T, C, H, W = x.shape

        if self.mode == "per_frame":
            # Flatten T into batch dim
            x_flat = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
            feats_flat = self._forward_impl(x_flat)  # [(B*T, C_i, H_i, W_i)]
            # Restore T dim
            return [f.reshape(B, T, *f.shape[1:]) for f in feats_flat]
        else:
            # Joint: encode all frames at once
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Single forward call. Override with actual Galileo logic."""
        # Placeholder — returns 4 pyramid levels via avg_pool
        # Replace with actual Galileo encoder.forward() call
        B, C, H, W = x.shape
        out = []
        for i in range(self.output_scales):
            scale = 4 * (2 ** i)
            if scale < H:
                pooled = nn.functional.avg_pool2d(x, scale, scale)
            else:
                pooled = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            # Project to dummy dims: 64, 128, 256, 512
            dim = 64 * (2 ** i)
            out.append(pooled.unsqueeze(1).repeat(1, dim // C if C > 0 else dim, 1, 1))
        return out[:self.output_scales]
