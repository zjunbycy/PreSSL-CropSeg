"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        mask = (target != self.ignore_index).unsqueeze(1).float()
        target_oh = target_oh * mask
        intersection = (probs * target_oh).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1 - pt) ** self.gamma
        mask = (target != self.ignore_index).float()
        loss = -self.alpha * focal_weight * log_pt * mask
        return loss.sum() / mask.sum().clamp(min=1)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 ignore_index: int = 255):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce_loss(logits, target) + \
               self.dice_weight * self.dice_loss(logits, target)


def build_loss(cfg: dict) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "ce")

    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=loss_cfg.get("ignore_index", 255))
    elif name == "dice":
        return DiceLoss(ignore_index=loss_cfg.get("ignore_index", 255))
    elif name == "focal":
        return FocalLoss(ignore_index=loss_cfg.get("ignore_index", 255))
    elif name == "combined":
        return CombinedLoss(
            ce_weight=loss_cfg.get("ce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            ignore_index=loss_cfg.get("ignore_index", 255),
        )
    raise ValueError(f"Unknown loss: {name}")
