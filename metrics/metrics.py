"""Evaluation metrics for semantic segmentation."""
import numpy as np
import torch
from typing import Dict, Optional
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """Computes IoU, mIoU, OA (overall accuracy), and per-class metrics.

    Accumulates predictions and targets across batches, then computes
    final metrics.

    Args:
        num_classes: number of classes
        ignore_index: label to ignore in evaluation
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Accumulate batch predictions.

        Args:
            pred: (B, H, W) or (B, C, H, W) predictions
            target: (B, H, W) ground truth
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.detach().cpu().numpy().flatten()
        target = target.detach().cpu().numpy().flatten()

        # Filter ignore_index
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        self.confusion += confusion_matrix(
            target, pred, labels=range(self.num_classes)
        )

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated confusion matrix.

        Returns:
            dict with OA, mIoU, IoU per class, and per-class accuracy
        """
        cm = self.confusion
        # Remove classes that never appear
        valid = cm.sum(axis=1) > 0
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            return {"OA": 0.0, "mIoU": 0.0}

        cm_valid = cm[valid][:, valid]

        # Overall accuracy
        oa = np.diag(cm_valid).sum() / cm_valid.sum().clip(min=1)

        # IoU per class
        intersection = np.diag(cm_valid)
        union = cm_valid.sum(axis=0) + cm_valid.sum(axis=1) - intersection
        iou = intersection / union.clip(min=1)

        miou = iou.mean()

        metrics = {
            "OA": float(oa),
            "mIoU": float(miou),
        }

        # Per-class IoU (use original indices)
        for idx in valid_indices:
            i = intersection[np.where(valid_indices == idx)[0][0]]
            u = union[np.where(valid_indices == idx)[0][0]]
            metrics[f"IoU_{idx}"] = float(i / u.clip(min=1))

        return metrics


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Quick single-batch mIoU computation.

    Args:
        pred: (B, C, H, W) logits or (B, H, W) class indices
        target: (B, H, W) class indices
        num_classes: number of classes

    Returns:
        mIoU value
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())

    return np.mean(ious) if ious else 0.0
