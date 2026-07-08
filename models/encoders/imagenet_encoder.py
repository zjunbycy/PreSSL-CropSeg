"""ImageNet-pretrained encoder via SMP for baseline comparison."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Optional


class ImageNetEncoder(nn.Module):
    """SMP encoder with ImageNet pretrained weights. Processes frames independently.

    Args:
        name: SMP encoder name (resnet50, efficientnet-b3, mit_b2, etc.)
        in_channels: S2 spectral bands
        weights: 'imagenet' or None
        depth: number of feature pyramid levels
        output_stride: encoder stride
    """

    def __init__(
        self,
        name: str = "resnet50",
        in_channels: int = 10,
        weights: Optional[str] = "imagenet",
        depth: int = 5,
        output_stride: int = 16,
        output_scales: int = 4,
        freeze: bool = False,
    ):
        super().__init__()
        self.name = name
        self.output_scales = output_scales
        self.freeze = freeze

        if weights is not None:
            self.encoder = smp.encoders.get_encoder(
                name, in_channels=3, depth=depth,
                weights=weights, output_stride=output_stride,
            )
            if in_channels != 3:
                self._adapt_first_conv(in_channels)
        else:
            self.encoder = smp.encoders.get_encoder(
                name, in_channels=in_channels, depth=depth,
                weights=None, output_stride=output_stride,
            )

        self._depth = depth
        self._out_channels = self.encoder.out_channels[-self.output_scales:]

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def _adapt_first_conv(self, in_channels: int):
        for attr in ['conv1', 'patch_embed', 'stem', 'conv_stem']:
            old_conv = getattr(self.encoder, attr, None)
            if old_conv is not None:
                break
        if old_conv is None:
            return

        old_weight = old_conv.weight
        out_ch, _, k_h, k_w = old_weight.shape
        new_conv = nn.Conv2d(
            in_channels, out_ch, kernel_size=(k_h, k_w),
            stride=old_conv.stride, padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data[:, :3] = old_weight.data
        mean_rgb = old_weight.data.mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            new_conv.weight.data[:, c] = mean_rgb.squeeze(1)
        setattr(self.encoder, attr, new_conv)

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Per-frame encoding.

        Args:
            x: (B, T, C, H, W)

        Returns:
            List of [(B, T, C_i, H_i, W_i), ...]
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feats_flat = self.encoder(x_flat)  # [(B*T, C_i, H_i, W_i)]
        feats_flat = feats_flat[-self.output_scales:]
        return [f.reshape(B, T, *f.shape[1:]) for f in feats_flat]
