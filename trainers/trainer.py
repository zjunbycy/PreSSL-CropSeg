"""Training and evaluation loops."""
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics.metrics import SegmentationMetrics
from data.augment import TemporalAugmentation


class Trainer:
    """Training loop with AMP, logging, checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        cfg: dict,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Logging
        log_cfg = cfg.get("logging", {})
        self.log_dir = os.path.join(
            log_cfg.get("log_dir", "logs"),
            log_cfg.get("experiment_name", "experiment"),
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.log_interval = log_cfg.get("log_interval", 10)
        self.val_interval = log_cfg.get("val_interval", 1)
        self.save_top_k = log_cfg.get("save_top_k", 3)

        # Training config
        train_cfg = cfg.get("training", {})
        self.max_epochs = train_cfg.get("num_epochs", 100)
        self.grad_clip = train_cfg.get("grad_clip", None)
        self.use_amp = train_cfg.get("amp", False)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Data
        self.num_classes = cfg["data"]["num_classes"]

        # Augmentation
        self.augment = TemporalAugmentation()

        # State
        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.best_epoch = 0
        self.checkpoints: list = []  # (miou, path) tuples

        # Metrics
        self.train_metrics = SegmentationMetrics(self.num_classes)
        self.val_metrics = SegmentationMetrics(self.num_classes)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            dict of average metrics
        """
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            (data, dates), target = batch

            # Cast to float and move to device
            if isinstance(data, dict):
                data = {k: v.float().to(self.device) for k, v in data.items()}
                # For now use S2 only; multi-modal support later
                data_tensor = data.get("S2", list(data.values())[0])
            else:
                data_tensor = data.float().to(self.device)

            if isinstance(dates, dict):
                dates_tensor = dates.get("S2", list(dates.values())[0]).to(self.device)
            else:
                dates_tensor = dates.to(self.device) if dates is not None else None

            target = target.long().to(self.device)

            # Augmentation
            if self.augment is not None:
                data_tensor, target = self.augment(data_tensor, target)

            # Forward
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.model(data_tensor, dates_tensor)
                loss = self.criterion(output, target)

            # Backward
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            total_loss += loss.item()
            self.train_metrics.update(output.detach(), target)

            # Log
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        # Epoch-level metrics
        metrics = self.train_metrics.compute()
        metrics["loss"] = total_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            dict of average metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        for batch in pbar:
            (data, dates), target = batch

            if isinstance(data, dict):
                data = {k: v.float().to(self.device) for k, v in data.items()}
                data_tensor = data.get("S2", list(data.values())[0])
            else:
                data_tensor = data.float().to(self.device)

            if isinstance(dates, dict):
                dates_tensor = dates.get("S2", list(dates.values())[0]).to(self.device)
            else:
                dates_tensor = dates.to(self.device) if dates is not None else None

            target = target.long().to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(data_tensor, dates_tensor)
                loss = self.criterion(output, target)

            total_loss += loss.item()
            self.val_metrics.update(output, target)

        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / len(self.val_loader)

        return metrics

    def fit(self):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Logs: {self.log_dir}")

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Log training
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            # Validate
            if epoch % self.val_interval == 0:
                val_metrics = self.val_epoch()

                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                val_miou = val_metrics.get("mIoU", 0.0)

                # Checkpoint
                self._save_checkpoint(val_miou)

                # Scheduler step (if ReduceLROnPlateau)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_miou)
                    else:
                        self.scheduler.step()

                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val mIoU: {val_miou:.4f} | "
                    f"Best: {self.best_val_miou:.4f} (epoch {self.best_epoch}) | "
                    f"Time: {elapsed:.1f}s"
                )
            else:
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

        print(f"Training complete. Best mIoU: {self.best_val_miou:.4f} at epoch {self.best_epoch}")
        self.writer.close()

    def _save_checkpoint(self, val_miou: float):
        """Save model checkpoint, keeping top-k by mIoU."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_miou": val_miou,
            "cfg": self.cfg,
        }

        path = os.path.join(self.log_dir, f"checkpoint_epoch{self.current_epoch}.pth")
        torch.save(checkpoint, path)

        self.checkpoints.append((val_miou, path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove worst checkpoints beyond top-k
        while len(self.checkpoints) > self.save_top_k:
            _, old_path = self.checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        if val_miou > self.best_val_miou:
            self.best_val_miou = val_miou
            self.best_epoch = self.current_epoch
            best_path = os.path.join(self.log_dir, "best_model.pth")
            torch.save(checkpoint, best_path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Standalone evaluation on a test set.

    Args:
        model: trained model
        data_loader: test dataloader
        device: device
        num_classes: number of classes

    Returns:
        dict of metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes)

    for batch in tqdm(data_loader, desc="Evaluating"):
        (data, dates), target = batch

        if isinstance(data, dict):
            data = {k: v.float().to(device) for k, v in data.items()}
            data_tensor = data.get("S2", list(data.values())[0])
        else:
            data_tensor = data.float().to(device)

        if isinstance(dates, dict):
            dates_tensor = dates.get("S2", list(dates.values())[0]).to(device)
        else:
            dates_tensor = dates.to(device) if dates is not None else None

        target = target.long().to(device)
        output = model(data_tensor, dates_tensor)
        metrics.update(output, target)

    return metrics.compute()
