# PreSSL-CropSeg

基于遥感自监督预训练模型 Galileo 的多时相耕地语义分割项目。项目以 PASTIS 为主要数据集，对比 ImageNet 监督预训练与 Galileo 遥感自监督预训练在作物/耕地分割上的迁移效果。

当前代码优先完成默认方案：**晚期融合 Late Fusion**。整体流程为：

```text
PASTIS Sentinel-2 time series
  -> shared per-frame encoder
  -> temporal-aware DPT decoder
  -> semantic mask logits
```

## 当前状态

- PASTIS 数据集目录已按代码约定放在 `data/PASTIS`。
- 本地 PASTIS 标签值为 `0..19`，因此默认 `num_classes=20`。
- Galileo 使用已转换好的 Hugging Face 权重：
  [BiliSakura/GALILEO-transformers](https://huggingface.co/BiliSakura/GALILEO-transformers)
- 默认 Galileo 权重子目录为 `galileo-base-patch8`，首次运行会下载到 `pretrained/galileo-base-patch8`。
- 已完成并 smoke test：
  - `configs/exp_imagenet_baseline.yaml`
  - `configs/exp_linear_probe.yaml`
  - `configs/exp_late.yaml` 的前向路径

## 环境搭建

推荐使用 Python 3.11。

```bash
conda create -n presl python=3.11 -y
conda activate presl
```

安装 CUDA 12.1 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import os; print(os.path.exists('data/PASTIS/metadata.geojson'))"
```

## 数据集

PASTIS 下载地址：

- https://zenodo.org/record/5012942

期望目录结构：

```text
PreSSL-CropSeg/
  data/
    PASTIS/
      metadata.geojson
      NORM_S2_patch.json
      DATA_S2/
        S2_10000.npy
        ...
      ANNOTATIONS/
        TARGET_10000.npy
        ...
```

数据输入形状：

```text
(B, T, 10, 128, 128)
```

其中 `T` 为变长时间序列，`10` 为 Sentinel-2 光谱波段数。

## 实验配置

所有实验配置位于 `configs/`。实验 yaml 只写变化部分，运行时会自动与 `configs/default.yaml` 合并。

| 配置文件 | 编码器 | 融合策略 | 编码器状态 | 用途 |
|---|---|---|---|---|
| `configs/exp_imagenet_baseline.yaml` | ImageNet ResNet50 | Late | 微调 | 首先运行，验证完整流程 |
| `configs/exp_linear_probe.yaml` | Galileo | Late | 冻结 | 测试 Galileo 特征质量 |
| `configs/exp_late.yaml` | Galileo | Late | 冻结 | 当前默认主方案 |
| `configs/exp_bottleneck.yaml` | Galileo | Bottleneck | 微调 | 中期融合对比 |
| `configs/exp_decision.yaml` | Galileo | Decision | 微调 | 决策融合消融 |

## 快速试跑

为了快速验证代码链路，可以限制 batch 数和时间步数：

```bash
python scripts/train.py ^
  --config configs/exp_imagenet_baseline.yaml ^
  --batch-size 1 ^
  --epochs 1 ^
  --num-workers 0 ^
  --max-train-batches 1 ^
  --max-val-batches 1 ^
  --max-timesteps 2
```

Galileo 线性探测建议先关闭 AMP 试跑：

```bash
python scripts/train.py ^
  --config configs/exp_linear_probe.yaml ^
  --batch-size 1 ^
  --epochs 1 ^
  --num-workers 0 ^
  --max-train-batches 1 ^
  --max-val-batches 1 ^
  --max-timesteps 2 ^
  --no-amp
```

## 正式训练

推荐顺序：

```bash
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --batch-size 1
python scripts/train.py --config configs/exp_linear_probe.yaml --batch-size 1 --no-amp
python scripts/train.py --config configs/exp_late.yaml --batch-size 1 --no-amp
python scripts/train.py --config configs/exp_bottleneck.yaml --batch-size 1
python scripts/train.py --config configs/exp_decision.yaml --batch-size 1
```

常用覆盖参数：

```bash
python scripts/train.py --config configs/exp_late.yaml --epochs 100
python scripts/train.py --config configs/exp_late.yaml --lr 0.0005
python scripts/train.py --config configs/exp_late.yaml --data-root "D:/PASTIS"
```

日志和 checkpoint 默认输出到：

```text
logs/<experiment_name>/
```

包含：

- TensorBoard event 文件
- `checkpoint_epoch{N}.pth`
- `best_model.pth`

查看 TensorBoard：

```bash
tensorboard --logdir logs
```

## 评估

验证集：

```bash
python scripts/eval.py ^
  --config configs/exp_imagenet_baseline.yaml ^
  --checkpoint logs/imagenet_late_fusion/best_model.pth ^
  --split val
```

测试集：

```bash
python scripts/eval.py ^
  --config configs/exp_imagenet_baseline.yaml ^
  --checkpoint logs/imagenet_late_fusion/best_model.pth ^
  --split test
```

## 模型结构

```text
Input: (B, T, 10, 128, 128)
  |
  +-- Encoder, shared across time
  |     +-- Galileo via Hugging Face transformers
  |     +-- ImageNet baseline via segmentation_models_pytorch
  |     -> [F1, F2, F3, F4], each keeps T
  |
  +-- Temporal fusion
  |     +-- Late fusion: temporal-aware DPT decoder
  |     +-- Bottleneck fusion: collapse T after encoder
  |     +-- Decision fusion: average per-frame logits
  |
  +-- Segmentation logits: (B, 20, 128, 128)
```

晚期融合是当前主线：编码器逐帧提取多尺度特征，DPT 解码阶段逐层进行时间维度聚合，最终输出统一空间分辨率的分割 logits。

## 项目结构

```text
configs/                 YAML 实验配置
data/                    PASTIS dataset、collate、augmentation
losses/                  CE / Dice / Focal / Combined loss
metrics/                 OA、mIoU、per-class IoU
models/
  encoders/              Galileo / ImageNet encoder
  fusion/                temporal fusion and DPT decoder
  temporal_seg_model.py  model factory and full assembly
scripts/
  train.py               training entry point
  eval.py                evaluation entry point
trainers/                training loop, AMP, checkpoint, TensorBoard
pretrained/              downloaded Galileo weights
logs/                    training outputs
```

## 注意事项

- 本地 PASTIS 标签为 `0..19`，不要把 `num_classes` 改回 19。
- Galileo 的真实权重依赖 `transformers`、`huggingface-hub` 和 `trust_remote_code=True`。
- Galileo + AMP 可能出现数值不稳定；线性探测和默认 late 方案建议先用 `--no-amp` 验证。
- 冻结 Galileo encoder 时，wrapper 会保持 encoder 为 eval 模式，只训练 decoder/head。
- 当前训练器是轻量 PyTorch loop。后续计划迁移到 Hugging Face `accelerate`/`Trainer`，见 `TODO.md`。

## 参考

- [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark)
- [GALILEO-transformers](https://huggingface.co/BiliSakura/GALILEO-transformers)
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
