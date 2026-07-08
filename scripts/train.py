"""Training entry point for temporal crop segmentation.

Usage:
  python scripts/train.py --config configs/exp_late.yaml
  python scripts/train.py --config configs/exp_imagenet_baseline.yaml
  python scripts/train.py --config configs/exp_linear_probe.yaml --lr 0.001
"""

import argparse
import os
import sys
import random

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pastis_dataset import PASTIS_Dataset
from data.collate import pad_collate
from models.temporal_seg_model import build_model
from losses.losses import build_loss
from trainers.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    set_seed(cfg["training"].get("seed", 42))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Encoder: {cfg['model']['encoder']['type']}")
    print(f"Fusion:  {cfg['model']['fusion_strategy']}")

    # Datasets
    data_cfg = cfg["data"]
    train_dataset = PASTIS_Dataset(
        folder=data_cfg["root"],
        norm=data_cfg.get("norm", True),
        target="semantic",
        folds=data_cfg.get("folds", [1, 2, 3]),
        reference_date=data_cfg.get("reference_date", "2018-09-01"),
        sats=data_cfg.get("sats", ["S2"]),
    )
    val_dataset = PASTIS_Dataset(
        folder=data_cfg["root"],
        norm=data_cfg.get("norm", True),
        target="semantic",
        folds=data_cfg.get("val_folds", [4]),
        reference_date=data_cfg.get("reference_date", "2018-09-01"),
        sats=data_cfg.get("sats", ["S2"]),
    )

    train_cfg = cfg["training"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate,
        pin_memory=True,
    )

    # Model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {n_params:.1f}M total, {trainable:.1f}M trainable")

    # Loss
    criterion = build_loss(cfg)

    # Optimizer
    if train_cfg.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            momentum=0.9,
        )

    # Scheduler
    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["num_epochs"],
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5,
        )
    else:
        scheduler = None

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed: {args.resume} (mIoU={ckpt.get('val_miou', 'N/A'):.4f})")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
