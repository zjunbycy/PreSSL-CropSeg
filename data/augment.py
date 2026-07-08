"""Data augmentation for temporal satellite images.

Spatial augmentations applied consistently across all time frames.
"""
import random
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


class TemporalAugmentation:
    """Apply same spatial transforms to every frame in a time series."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotation: bool = True,
        rotation_degrees: int = 90,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation = rotation
        self.rotation_degrees = rotation_degrees

    def __call__(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (T, C, H, W) image time series
            target: (H, W) label map

        Returns:
            Augmented (data, target)
        """
        # Horizontal flip
        if random.random() < self.hflip_prob:
            data = torch.flip(data, dims=[-1])
            target = torch.flip(target, dims=[-1])

        # Vertical flip
        if random.random() < self.vflip_prob:
            data = torch.flip(data, dims=[-2])
            target = torch.flip(target, dims=[-2])

        # Rotation by multiples of 90 degrees
        if self.rotation:
            k = random.randint(0, 3)
            if k > 0:
                data = torch.rot90(data, k, dims=[-2, -1])
                target = torch.rot90(target, k, dims=[-2, -1])

        return data, target


class MonoDateAugmentation:
    """Minimal augmentation that works on single-date inputs."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.hflip_prob:
            data = torch.flip(data, dims=[-1])
            target = torch.flip(target, dims=[-1])

        if random.random() < self.vflip_prob:
            data = torch.flip(data, dims=[-2])
            target = torch.flip(target, dims=[-2])

        return data, target
