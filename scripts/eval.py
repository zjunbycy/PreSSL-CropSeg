"""Evaluation entry point."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pastis_dataset import PASTIS_Dataset
from data.collate import pad_collate
from models.temporal_seg_model import build_model
from trainers.trainer import evaluate
from scripts.train import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temporal crop segmentation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test", "all"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0, cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    if args.data_root:
        cfg["data"]["root"] = args.data_root

    # Device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset
    data_cfg = cfg["data"]
    if args.split == "val":
        folds = data_cfg.get("val_folds", [4])
    elif args.split == "test":
        folds = data_cfg.get("test_folds", [5])
    else:
        folds = data_cfg.get("val_folds", [4]) + data_cfg.get("test_folds", [5])

    dataset = PASTIS_Dataset(
        folder=data_cfg["root"],
        norm=data_cfg.get("norm", True),
        target=data_cfg.get("target", "semantic"),
        folds=folds,
        reference_date=data_cfg.get("reference_date", "2018-09-01"),
        mono_date=data_cfg.get("mono_date", None),
        sats=data_cfg.get("sats", ["S2"]),
        cache=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
    )

    # Model
    model = build_model(cfg)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    val_miou = checkpoint.get("val_miou", None)
    if isinstance(val_miou, (float, int)):
        print(f"  Val mIoU: {val_miou:.4f}")
    else:
        print("  Val mIoU: N/A")

    model.to(device)

    # Evaluate
    metrics = evaluate(model, loader, device, data_cfg["num_classes"])

    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({args.split} split, {len(dataset)} samples)")
    print("=" * 50)
    print(f"Overall Accuracy (OA): {metrics['OA']:.4f}")
    print(f"Mean IoU (mIoU):     {metrics['mIoU']:.4f}")
    print("-" * 50)
    print("Per-class IoU:")
    for k, v in sorted(metrics.items()):
        if k.startswith("IoU_"):
            print(f"  Class {k.replace('IoU_', ''):>3s}: {v:.4f}")


if __name__ == "__main__":
    main()
