"""
方案二: Late Fusion — 3D-Aware DPT Decoder

Core idea: instead of collapsing T before DPT, keep T inside DPT and
use 3D convolution or temporal attention at each decoder stage to
gradually fold temporal information into spatial features.

The T dimension is absorbed layer by layer, not all at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# 3D Temporal Collapse Modules
# ──────────────────────────────────────────────────────────────────────

class TemporalCollapse3DConv(nn.Module):
    """3D convolution along T axis to reduce temporal dimension.

    Args:
        channels: feature channels at this scale
        kernel_t: temporal kernel size (how many adjacent frames to fuse)
        stride_t: temporal stride (reduce factor)
    """

    def __init__(self, channels: int, kernel_t: int = 3, stride_t: int = 2):
        super().__init__()
        padding_t = kernel_t // 2
        self.conv3d = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_t, 1, 1),
            stride=(stride_t, 1, 1),
            padding=(padding_t, 0, 0),
            bias=False,
        )
        self.norm = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) or (B, C, H, W) if T already 1

        Returns:
            (B, T', C, H, W) with T' < T
        """
        if x.dim() == 4:
            return x.unsqueeze(1)  # (B, 1, C, H, W) — no more collapse needed

        B, T, C, H, W = x.shape
        if T == 1:
            return x

        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.act(x)
        # (B, C, T', H, W) → (B, T', C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class TemporalCollapseAttention(nn.Module):
    """Temporal self-attention to merge T frames into 1.

    Uses a learnable [CLS_T] token that cross-attends to all T frames.
    More expressive than 3D conv, slightly more params.

    Args:
        channels: feature channels
        num_heads: attention heads
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, channels, 1, 1))
        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) or (B, C, H, W)

        Returns:
            (B, 1, C, H, W) — T collapsed to 1
        """
        if x.dim() == 4:
            return x.unsqueeze(1)

        B, T, C, H, W = x.shape
        if T == 1:
            return x

        # Reshape: (B, T, C, H, W) → (B*H*W, T, C)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        # [CLS] token per spatial position
        cls_token = self.temporal_cls.expand(B * H * W, 1, C)  # (B*H*W, 1, C)
        x_with_cls = torch.cat([cls_token, x_flat], dim=1)  # (B*H*W, 1+T, C)

        out, _ = self.attn(x_with_cls, x_with_cls, x_with_cls)

        # Take [CLS] token only
        out = out[:, 0]  # (B*H*W, C)
        out = self.norm(out)

        # Reshape back
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return out.unsqueeze(1)  # (B, 1, C, H, W)


# ──────────────────────────────────────────────────────────────────────
# DPT Components (re-implemented for T-awareness)
# ──────────────────────────────────────────────────────────────────────

class ReassembleBlock(nn.Module):
    """Project encoder features to uniform dim, optionally with T collapse."""

    def __init__(self, in_ch: int, out_ch: int, scale: int, collapse_t: bool = True):
        super().__init__()
        self.scale = scale
        self.collapse_t = collapse_t

        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if scale > 1:
            self.resize = nn.ConvTranspose2d(out_ch, out_ch,
                                             kernel_size=scale, stride=scale)
        else:
            self.resize = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) or (B, C, H, W)"""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            if self.collapse_t and T > 1:
                # Simple mean collapse before projection
                x = x.mean(dim=1)  # (B, C, H, W)
            elif T == 1:
                x = x.squeeze(1)
            else:
                # Flatten T into batch for spatial processing
                x = x.reshape(B * T, C, H, W)
                x = self.proj(x)
                x = self.resize(x)
                _, _, H2, W2 = x.shape
                x = x.reshape(B, T, -1, H2, W2)
                return x

        x = self.proj(x)
        x = self.resize(x)
        return x


class ResidualConvUnit(nn.Module):
    """RCU: two 3x3 convs with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class FusionBlock(nn.Module):
    """Multi-scale feature fusion stage in DPT."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.res_conv1 = ResidualConvUnit(channels)
        self.res_conv2 = ResidualConvUnit(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads,
                                          batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of (B, C, H, W) at different scales, sorted by resolution
                      smallest → largest

        Returns:
            (B, C, H, W) fused feature at highest resolution
        """
        # Start from the coarsest feature
        x = features[0]
        x = self.res_conv1(x)

        for f in features[1:]:
            # Upsample x to match f
            if x.shape[-2:] != f.shape[-2:]:
                x = F.interpolate(x, size=f.shape[-2:], mode='bilinear',
                                  align_corners=False)
            x = x + f
            x = self.res_conv2(x)

        # Global self-attention at highest scale
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm(x_attn + x_flat).transpose(1, 2).reshape(B, C, H, W)

        return x


# ──────────────────────────────────────────────────────────────────────
# 3D-Aware DPT Decoder (方案二: Late Fusion)
# ──────────────────────────────────────────────────────────────────────

class TemporalAwareDPTDecoder(nn.Module):
    """DPT decoder that gradually collapses T at each reassemble stage.

    Strategy: at each Reassemble level, apply TemporalCollapse before
    spatial fusion. Coarser scales collapse T earlier; finer scales
    collapse T later (or not at all before the final fusion).

    Args:
        encoder_channels: [C1, C2, C3, C4] from encoder
        decoder_channels: unified feature dim for DPT (default 256)
        num_classes: output classes (19 for PASTIS)
        collapse_schedule: how many 3D conv stages per scale (e.g. [3, 2, 1, 0])
                           more stages → more aggressive T reduction at that scale
        temporal_module: 'conv3d' or 'attention'
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: int = 256,
        num_classes: int = 19,
        collapse_schedule: Optional[List[int]] = None,
        temporal_module: str = "attention",
    ):
        super().__init__()
        self.decoder_channels = decoder_channels

        n_scales = len(encoder_channels)
        if collapse_schedule is None:
            collapse_schedule = [3, 2, 1, 0][:n_scales]
        self.collapse_schedule = collapse_schedule

        # Reassemble blocks: project each encoder scale → decoder_channels
        scales = [4, 8, 16, 32][:n_scales]
        self.reassembles = nn.ModuleList([
            ReassembleBlock(ec, decoder_channels, s)
            for ec, s in zip(encoder_channels, scales)
        ])

        # Temporal collapse per scale (can be identity if collapse_schedule[i]==0)
        CollapseCls = TemporalCollapseAttention if temporal_module == "attention" \
                      else TemporalCollapse3DConv
        self.collapses = nn.ModuleList()
        for ec, n_collapse in zip(encoder_channels, collapse_schedule):
            seq = []
            for _ in range(n_collapse):
                seq.append(CollapseCls(ec))
            self.collapses.append(nn.Sequential(*seq) if seq else nn.Identity())

        # DPT fusion block
        self.fusion = FusionBlock(decoder_channels)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, num_classes, 1),
        )

    def forward(
        self,
        features: List[torch.Tensor],  # [F1..F4], each (B, T, C_i, H_i, W_i)
    ) -> torch.Tensor:
        """
        Returns:
            (B, num_classes, H, W) logits at input resolution
        """
        reassembled = []

        for i, feat in enumerate(features):
            # Step 1: Gradually collapse T via 3D conv/attention at each scale
            f = self.collapses[i](feat)  # (B, T', C_i, H_i, W_i) with smaller T'
            # Step 2: Reassemble projects to uniform dim + upsamples
            f = self.reassembles[i](f)  # (B, decoder_channels, H_i', W_i')
            reassembled.append(f)

        # Step 3: FPN-style fusion
        fused = self.fusion(reassembled)  # (B, decoder_channels, H, W)

        # Step 4: Segmentation logits
        logits = self.seg_head(fused)  # (B, num_classes, H, W)

        return logits
