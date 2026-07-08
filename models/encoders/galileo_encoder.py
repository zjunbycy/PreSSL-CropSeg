"""Galileo Encoder Wrapper — loads BiliSakura/GALILEO-transformers weights.

Galileo is a multimodal remote sensing foundation model. It needs:
  space_time_x: (B, H, W, T, 13) — S1(2) + S2(10) + NDVI(1)
  space_x:      (B, H, W, 16)   — SRTM + DW + WC
  time_x:       (B, T, 6)       — ERA5 + TC + VIIRS
  static_x:     (B, 18)         — LANDSCAN + location + DW_static + WC_static
  + masks for each modality group
  + months: (B, T) long

For PASTIS (S2-only, 10 bands), we:
  - Fill S2 bands into space_time_x, compute NDVI from B8 & B4
  - Set S1 bands to 0 with mask=1 (missing)
  - Set all other modalities to 0 with mask=1 (missing)
  - Set months from PASTIS dates + reference_date

Architecture: GalileoEncoderModel → last_hidden_state (B, N_patches, 768)
We reshape N_patches to spatial grid for multi-scale feature pyramid.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

# Band indices in space_time_x (13 bands total):
#   S1: 0-VV, 1-VH
#   S2: 2-B2, 3-B3, 4-B4, 5-B5, 6-B6, 7-B7, 8-B8, 9-B8A, 10-B11, 11-B12
#   NDVI: 12
# But PASTIS S2 has only 10 bands (no S1): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
# S2 index in space_time_x: positions 2-11 → ordered same as PASTIS

S2_BANDS_IN_SPACETIME = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # B2..B12
S2_B8_IDX = 8   # B8 in space_time_x
S2_B4_IDX = 4   # B4 in space_time_x
SPACETIME_CHANNELS = 13
SPACE_CHANNELS = 16
TIME_CHANNELS = 6
STATIC_CHANNELS = 18
NUM_SPACETIME_GROUPS = 7  # S1, S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m, S2_SWIR, NDVI
NUM_SPACE_GROUPS = 3      # SRTM, DW, WC
NUM_TIME_GROUPS = 3       # ERA5, TC, VIIRS
NUM_STATIC_GROUPS = 4     # LS, location, DW_static, WC_static

# Which group each S2 band belongs to (positions in space_time_mask)
S2_GROUP_MASK_IDX = [1, 1, 1, 2, 2, 2, 3, 4, 5, 5]  # per S2 band in order


class _PlaceholderGalileoEncoder(nn.Module):
    """Small local fallback used when Galileo weights or dependencies are absent."""

    def __init__(self, in_channels: int = 10, output_scales: int = 4):
        super().__init__()
        channels = [64, 128, 256, 512][:output_scales]
        self.stages = nn.ModuleList()

        prev = in_channels
        for i, ch in enumerate(channels):
            stride = 4 if i == 0 else 2
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(prev, ch, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                )
            )
            prev = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def _download_hf_subfolder(model_name: str, subfolder: str, local_dir: str) -> str:
    """Download a subfolder model from HF Hub to local directory."""
    from huggingface_hub import snapshot_download
    import shutil

    snapshot_download(
        repo_id=model_name,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    # Move files from nested subfolder up
    nested = os.path.join(local_dir, subfolder)
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            src, dst = os.path.join(nested, f), os.path.join(local_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)
        shutil.rmtree(os.path.join(local_dir, ".cache"), ignore_errors=True)
    return local_dir


class GalileoEncoder(nn.Module):
    """Wrapper for Galileo pretrained encoder.

    Args:
        model_name: HF repo id
        subfolder: subfolder in the repo (galileo-base-patch8 / nano / tiny)
        mode: 'per_frame' — each frame independently; 'joint' — all at once
        in_channels: S2 bands (10 for PASTIS)
        img_size: spatial size (128)
        freeze: freeze encoder weights
        output_scales: how many feature scales to return (1-4)
    """

    def __init__(
        self,
        model_name: str = "BiliSakura/GALILEO-transformers",
        subfolder: str = "galileo-base-patch8",
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
        self.freeze = freeze

        local_path = os.path.join("pretrained", subfolder)
        self._using_placeholder = False
        try:
            if not os.path.exists(os.path.join(local_path, "config.json")):
                print(f"[Galileo] Downloading {subfolder} from {model_name} ...")
                _download_hf_subfolder(model_name, subfolder, local_path)

            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            print(f"[Galileo] Loaded {subfolder} from {local_path}")
        except Exception as e:
            print(f"[Galileo] Could not load {subfolder}: {e}")
            print(f"[Galileo] Using placeholder")
            self.encoder = self._build_placeholder()
            self._using_placeholder = True

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            self.encoder.train()

        self._out_channels = None
        self._patch_size = 8

    def _build_placeholder(self):
        return _PlaceholderGalileoEncoder(self.in_channels, self.output_scales)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    @property
    def out_channels(self) -> List[int]:
        if self._out_channels is not None:
            return self._out_channels
        dummy = torch.randn(1, 2, self.in_channels, self.img_size, self.img_size)
        with torch.no_grad():
            feats = self.forward(dummy)
        self._out_channels = [f.shape[2] for f in feats]
        return self._out_channels

    def _build_galileo_inputs(self, x: torch.Tensor, months: torch.Tensor = None):
        """Convert (B, C, H, W) tensor to Galileo's multimodal input dict.

        Args:
            x: (B, C, H, W) — S2 10-band image
            months: (B,) long, month values

        Returns:
            dict with space_time_x, space_x, time_x, static_x, masks, months, patch_size
        """
        B, C, H, W = x.shape
        device = x.device
        T = 1  # per-frame mode

        # --- space_time_x: (B, H, W, T, 13) ---
        # Fill S2 bands (10 bands → indices 2-11 in space_time)
        s_t_x = torch.zeros(B, H, W, T, SPACETIME_CHANNELS, device=device, dtype=x.dtype)
        for i, idx in enumerate(S2_BANDS_IN_SPACETIME):
            s_t_x[:, :, :, 0, idx] = x[:, i]  # x[:, i] is (B, H, W)

        # Compute NDVI: (B8 - B4) / (B8 + B4 + 1e-6)
        b8 = x[:, 6]   # B8 is 7th PASTIS band (index 6 in 0-based 10-band)
        b4 = x[:, 2]   # B4 is 3rd PASTIS band
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)  # (B, H, W)
        s_t_x[:, :, :, 0, 12] = ndvi

        # --- space_time_mask: (B, H, W, T, 7 groups) ---
        # S1 group (idx 0): mask=1 (missing)
        # S2 groups (idx 1-5): mask=0 (present)
        # NDVI group (idx 6): mask=0 (present)
        s_t_m = torch.ones(B, H, W, T, NUM_SPACETIME_GROUPS, device=device, dtype=x.dtype)
        s_t_m[:, :, :, 0, 1] = 0  # S2_RGB
        s_t_m[:, :, :, 0, 2] = 0  # S2_Red_Edge
        s_t_m[:, :, :, 0, 3] = 0  # S2_NIR_10m
        s_t_m[:, :, :, 0, 4] = 0  # S2_NIR_20m
        s_t_m[:, :, :, 0, 5] = 0  # S2_SWIR
        s_t_m[:, :, :, 0, 6] = 0  # NDVI

        # --- space_x: (B, H, W, 16) all zero, mask all 1 ---
        sp_x = torch.zeros(B, H, W, SPACE_CHANNELS, device=device, dtype=x.dtype)
        sp_m = torch.ones(B, H, W, NUM_SPACE_GROUPS, device=device, dtype=x.dtype)

        # --- time_x: (B, T, 6) all zero, mask all 1 ---
        t_x = torch.zeros(B, T, TIME_CHANNELS, device=device, dtype=x.dtype)
        t_m = torch.ones(B, T, NUM_TIME_GROUPS, device=device, dtype=x.dtype)

        # --- static_x: (B, 18) all zero, mask all 1 ---
        st_x = torch.zeros(B, STATIC_CHANNELS, device=device, dtype=x.dtype)
        st_m = torch.ones(B, NUM_STATIC_GROUPS, device=device, dtype=x.dtype)

        # --- months: (B, T) ---
        if months is None:
            months = torch.ones(B, T, device=device, dtype=torch.long) * 6
        elif months.dim() == 1:
            months = months.unsqueeze(-1)  # (B,) → (B, 1)

        return {
            "space_time_x": s_t_x,
            "space_x": sp_x,
            "time_x": t_x,
            "static_x": st_x,
            "space_time_mask": s_t_m,
            "space_mask": sp_m,
            "time_mask": t_m,
            "static_mask": st_m,
            "months": months,
            "patch_size": self._patch_size,
        }

    def forward(self, x: torch.Tensor, months: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) PASTIS format
            months: (B, T) optional month values (1-12)

        Returns:
            [F1..F4], each (B, T, C_i, H_i, W_i)
        """
        B, T_frames, C, H, W = x.shape

        all_feats = []
        for t in range(T_frames):
            x_t = x[:, t]  # (B, C, H, W)
            m_t = months[:, t] if months is not None else None
            inputs = self._build_galileo_inputs(x_t, m_t)
            feats = self._forward_impl(inputs)  # [(B, C_i, H_i, W_i)]
            all_feats.append(feats)

        return [torch.stack([f[i] for f in all_feats], dim=1) for i in range(len(all_feats[0]))]

    def _forward_impl(self, galileo_inputs: dict) -> List[torch.Tensor]:
        """Forward through Galileo, extract multi-scale feature pyramid."""
        if self._using_placeholder:
            # Reconstruct PASTIS S2 bands from the Galileo-style input tensor.
            s_t = galileo_inputs["space_time_x"]  # (B, H, W, 1, 13)
            x_2d = s_t[:, :, :, 0, S2_BANDS_IN_SPACETIME]
            x_2d = x_2d.permute(0, 3, 1, 2).contiguous()  # (B, 10, H, W)
            return self.encoder(x_2d)

        # Real Galileo model
        output = self.encoder(**galileo_inputs)
        lhs = output.last_hidden_state  # (B, n_tokens, hidden_dim)
        B, n_tokens, hidden_dim = lhs.shape

        # Drop CLS, use all patch tokens. Galileo output mixes multiple modalities
        # so n_patches is not always a perfect square. Use adaptive pooling.
        patches_flat = lhs[:, 1:, :]  # (B, n_patches, D)

        features = []
        for i in range(self.output_scales):
            target = max(self.img_size // (4 * (2 ** i)), 1)
            n = target * target
            if n != patches_flat.shape[1]:
                pooled = F.adaptive_avg_pool1d(
                    patches_flat.transpose(1, 2), n
                ).transpose(1, 2)  # (B, n, D)
            else:
                pooled = patches_flat
            f = pooled.transpose(1, 2).reshape(B, hidden_dim, target, target)
            features.append(f)

        return features
