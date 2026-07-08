# PreSSL-CropSeg

Pre-trained Self-Supervised Learning model for temporal remote sensing **cropland semantic segmentation**, built on the [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark) and [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## Architecture

```
Input: Sentinel-2 Time Series (T × C × H × W)
  │
  ├─► Shared Encoder (per-frame, weights from SMP / SSL pretrained)
  │     └─► Multi-scale features per frame: [f₀, f₁, f₂, f₃, f₄]
  │
  ├─► Temporal Fusion (across T frames, per scale)
  │     └─► Fused multi-scale features: [F₀, F₁, F₂, F₃, F₄]
  │
  └─► Decoder (SMP: DeepLabV3+ / FPN / UPerNet / DPT)
        └─► Output: (num_classes × H × W)
```

Key insight: 2D pretrained backbones process each time frame independently with shared weights. Temporal information is fused at the feature level before the decoder.

## Project Structure

```
├── configs/            # YAML config files
├── data/               # Dataset, dataloader, augmentations
│   ├── pastis_dataset.py   # PASTIS PyTorch Dataset (from official)
│   ├── collate.py          # pad_collate for variable-length sequences
│   └── augment.py          # Augmentation pipeline
├── models/             # Model components
│   ├── encoder.py          # SMP backbone wrapper
│   ├── temporal_fusion.py  # Temporal fusion modules
│   ├── decoder.py          # SMP decoder wrapper
│   └── temporal_seg_model.py  # Full model assembly
├── losses/             # Loss functions
├── metrics/            # Evaluation metrics (IoU, mIoU, PASTIS panoptic)
├── trainers/           # Training & evaluation loops
├── scripts/            # Entry points
│   ├── train.py
│   └── eval.py
└── notebooks/          # Exploration & visualization
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download PASTIS dataset from Zenodo (29GB)
# https://zenodo.org/record/5012942

# Train
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/eval.py --config configs/default.yaml --checkpoint path/to/ckpt.pth
```

## Reference

- [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark) (ICCV 2021)
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [ZJU-GISLAB-COURSE-2026](https://github.com/Bili-Sakura/ZJU-GISLAB-COURSE-2026)
