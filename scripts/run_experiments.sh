#!/bin/bash
# Run all experiments for the 3-week project
# Requires PASTIS dataset extracted at data/PASTIS/

set -e
echo "=== PreSSL-CropSeg: Experiment Suite ==="

# ─── Experiment 1: Galileo + Late Fusion (Main Method) ───
echo "[1/5] Galileo Late Fusion (Main)"
python scripts/train.py --config configs/exp_late.yaml

# ─── Experiment 2: Galileo + Bottleneck Fusion ───
echo "[2/5] Galileo Bottleneck Fusion"
python scripts/train.py --config configs/exp_bottleneck.yaml

# ─── Experiment 3: Galileo + Decision Fusion ───
echo "[3/5] Galileo Decision Fusion"
python scripts/train.py --config configs/exp_decision.yaml

# ─── Experiment 4: ImageNet Baseline ───
echo "[4/5] ImageNet Late Fusion Baseline"
python scripts/train.py --config configs/exp_imagenet_baseline.yaml

# ─── Experiment 5: Galileo Linear Probe ───
echo "[5/5] Galileo Linear Probe"
python scripts/train.py --config configs/exp_linear_probe.yaml

echo "=== All experiments complete ==="
echo "Compare mIoU in logs/ for each experiment"
