"""
Full model assemblies for three fusion strategies.

方案一 (Bottleneck/Mid): Encoder → BottleneckFusion → Standard DPT
方案二 (Late):         Encoder → TemporalAwareDPTDecoder (3D collapse inside)
方案三 (Decision):     Encoder → Per-frame DPT → DecisionFusion (soft voting)

Config-driven factory: build_model(cfg) returns the right model.
"""

import torch
import torch.nn as nn
from typing import Optional

from models.encoders import GalileoEncoder, ImageNetEncoder
from models.fusion.fusion import BottleneckFusion, DecisionFusion
from models.fusion.temporal_dpt_decoder import TemporalAwareDPTDecoder


class PretrainedDPTDecoder(nn.Module):
    """Standard DPT decoder (no temporal handling). Used by 方案一 and 方案三."""

    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: int = 256,
        num_classes: int = 19,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels

        from models.fusion.temporal_dpt_decoder import (
            ReassembleBlock, FusionBlock, ResidualConvUnit,
        )

        scales = [4, 8, 16, 32][:len(encoder_channels)]
        self.reassembles = nn.ModuleList([
            ReassembleBlock(ec, decoder_channels, s)
            for ec, s in zip(encoder_channels, scales)
        ])
        self.fusion = FusionBlock(decoder_channels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, num_classes, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        """features: list of (B, C_i, H_i, W_i) — T already collapsed"""
        reassembled = [self.reassembles[i](f) for i, f in enumerate(features)]
        fused = self.fusion(reassembled)
        return self.seg_head(fused)


# ──────────────────────────────────────────────────────────────────────
# Model Registry & Factory
# ──────────────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "galileo": GalileoEncoder,
    "imagenet": ImageNetEncoder,
}


def _dates_to_months(dates: torch.Tensor, ref_year: int = 2018) -> torch.Tensor:
    """Convert day-offset dates to month numbers (1-12).

    Args:
        dates: (B, T) — days since reference_date (e.g. 2018-09-01)
        ref_year: year of the reference date

    Returns:
        (B, T) — month numbers 1-12
    """
    import datetime as dt
    ref_date = dt.date(ref_year, 9, 1)
    months = torch.zeros_like(dates, dtype=torch.long)
    for b in range(dates.shape[0]):
        for t in range(dates.shape[1]):
            d = dates[b, t].item()
            actual = ref_date + dt.timedelta(days=int(d))
            months[b, t] = actual.month
    return months


class TemporalSegModel(nn.Module):
    """Unified model supporting all three fusion strategies.

    Config keys used:
        model.encoder.type: "galileo" | "imagenet"
        model.encoder.*: passed to encoder constructor
        model.fusion_strategy: "bottleneck" | "late" | "decision"
        model.decoder_channels: int (256)
        data.num_classes: int (19)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        # ── Encoder ──
        enc_type = model_cfg.get("encoder", {}).get("type", "galileo")
        enc_kwargs = {k: v for k, v in model_cfg.get("encoder", {}).items()
                      if k != "type"}
        self.encoder = ENCODER_REGISTRY[enc_type](**enc_kwargs)
        self.encoder_type = enc_type

        # ── Fusion strategy ──
        self.fusion_strategy = model_cfg.get("fusion_strategy", "late")
        self.num_classes = data_cfg["num_classes"]

        # ── Init all attributes before eager build ──
        self._encoder_channels = None
        self._decoder_ch = model_cfg.get("decoder_channels", 256)
        self._temp_mod = model_cfg.get("temporal_module", "attention")
        self._collapse_sched = model_cfg.get("collapse_schedule", None)
        self.bottleneck_fusion = None
        self.decoder = None
        self.decision_fusion = DecisionFusion() if self.fusion_strategy == "decision" else None

        # ── Eagerly probe encoder to build decoder ──
        dummy = torch.randn(1, 2, 10, 128, 128)
        with torch.no_grad():
            features = self.encoder(dummy)
        self._ensure_built(features)

    def _ensure_built(self, features: list):
        """Build fusion/decoder modules after seeing encoder output dims."""
        if self._encoder_channels is not None:
            return

        channels = [f.shape[2] for f in features]
        self._encoder_channels = channels

        if self.fusion_strategy == "bottleneck":
            self.bottleneck_fusion = BottleneckFusion(channels)
            self.decoder = PretrainedDPTDecoder(
                channels, self._decoder_ch, self.num_classes,
            )
        elif self.fusion_strategy == "late":
            self.decoder = TemporalAwareDPTDecoder(
                channels, self._decoder_ch, self.num_classes,
                collapse_schedule=self._collapse_sched,
                temporal_module=self._temp_mod,
            )
        elif self.fusion_strategy == "decision":
            self.decoder = PretrainedDPTDecoder(
                channels, self._decoder_ch, self.num_classes,
            )

    def forward(
        self,
        x: torch.Tensor,
        dates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
            dates: (B, T) acquisition dates as day offsets from reference

        Returns:
            (B, num_classes, H, W) logits
        """
        # ── Convert dates to months for Galileo ──
        months = None
        if dates is not None and self.encoder_type == "galileo":
            months = _dates_to_months(dates)

        # ── Encoder ──
        if self.encoder_type == "galileo":
            features = self.encoder(x, months=months)
        else:
            features = self.encoder(x)

        self._ensure_built(features)

        # ── Fusion + Decoder ──
        if self.fusion_strategy == "bottleneck":
            spatial_feats = self.bottleneck_fusion(features, dates)
            return self.decoder(spatial_feats)

        elif self.fusion_strategy == "late":
            return self.decoder(features)

        elif self.fusion_strategy == "decision":
            B, T = x.shape[0], x.shape[1]
            per_frame_logits = []
            for t in range(T):
                feat_t = [f[:, t] for f in features]
                logit_t = self.decoder(feat_t)
                per_frame_logits.append(logit_t)
            stacked = torch.stack(per_frame_logits, dim=1)
            return self.decision_fusion(stacked)

        raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")


def build_model(cfg: dict) -> TemporalSegModel:
    """Factory: create TemporalSegModel from config dict."""
    return TemporalSegModel(cfg)
