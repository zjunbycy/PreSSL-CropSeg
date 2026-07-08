# PreSSL-CropSeg

基于自监督预训练（Galileo, ICML 2025）的时间序列遥感**耕地语义分割**项目，使用 [PASTIS 基准数据集](https://github.com/VSainteuf/pastis-benchmark)（ICCV 2021）。ZJU-GISLAB-COURSE-2026 课程项目。

## 研究动机

核心问题：**针对耕地分割任务，遥感领域自监督预训练（Galileo）相比标准 ImageNet 预训练能带来多大提升？**

PASTIS 数据集：2,433 个 Sentinel-2 时间序列样本，128×128 像素，10 个光谱波段，每样本 38–61 个时间步，18 种作物 + 背景 = 19 类。

---

## 环境搭建（首次使用）

### 1. 创建 conda 环境

```bash
conda create -n presl python=3.11 -y
conda activate presl
```

### 2. 安装 PyTorch（CUDA 12.1）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装其他依赖

```bash
cd PreSSL-CropSeg
pip install -r requirements.txt
```

### 4. 下载 PASTIS 数据集

从 Zenodo 下载并解压到项目 `data/` 目录：
- https://zenodo.org/record/5012942

最终目录结构应为：

```
PreSSL-CropSeg/
├── data/
│   └── PASTIS/
│       ├── metadata.geojson
│       ├── NORM_S2_patch.json
│       ├── DATA_S2/          # S2_10000.npy, ...
│       └── ANNOTATIONS/      # TARGET_10000.npy, ...
├── configs/
├── models/
├── scripts/
└── ...
```

### 5. 验证环境

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import os; print('PASTIS OK:', os.path.exists('data/PASTIS/metadata.geojson'))"
```

两个都输出 `True` 即环境就绪。

---

## 显存要求

| GPU | 建议 batch_size |
|---|---|
| RTX 3060 (6GB) | 1 |
| RTX 3080 (10GB) | 2 |
| RTX 4070+ (12GB+) | 4 |
| A100 / 4090 (24GB+) | 8 |

显存不够时减小 `--batch-size`：
```bash
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --batch-size 1
```

---

## 训练指南

### 配置文件说明

所有实验配置在 `configs/` 目录下。实验配置（`exp_*.yaml`）只写变化部分，运行时会自动与 `configs/default.yaml` 合并。

| 配置文件 | 编码器 | 融合策略 | 冻结编码器 | 目的 |
|---|---|---|---|---|
| `configs/exp_imagenet_baseline.yaml` | ImageNet ResNet50 | 晚期融合 | 否 | **推荐首先跑这个**，验证环境 |
| `configs/exp_late.yaml` | Galileo | 晚期融合 | 是 | 默认/主要配置 |
| `configs/exp_bottleneck.yaml` | Galileo | 瓶颈融合 | 否 | 中期融合对比实验 |
| `configs/exp_decision.yaml` | Galileo | 决策融合 | 否 | 无时间交互消融下界 |
| `configs/exp_linear_probe.yaml` | Galileo | 晚期融合 | 是 | SSL 纯特征质量测试 |

### 常用命令行覆盖

不用改 yaml 文件，直接传参覆盖：

```bash
# 减小 batch size 适配小显存
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --batch-size 1

# 改学习率
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --lr 0.001

# 改训练轮数
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --epochs 100

# 改数据路径
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --data-root "D:/PASTIS"

# 组合使用
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --batch-size 2 --epochs 50 --lr 0.0005
```

### 推荐训练顺序

| 优先级 | 命令 | 说明 |
|---|---|---|
| **1 先跑** | `python scripts/train.py --config configs/exp_imagenet_baseline.yaml` | ImageNet 基线，验证全流程 |
| **2 核心** | `python scripts/train.py --config configs/exp_linear_probe.yaml` | Galileo 线性探测 |
| **3 核心** | `python scripts/train.py --config configs/exp_late.yaml` | Galileo 微调，主方法 |
| **4 对比** | `python scripts/train.py --config configs/exp_bottleneck.yaml` | 瓶颈融合 |
| **5 消融** | `python scripts/train.py --config configs/exp_decision.yaml` | 决策融合（下界） |

### 监控训练

```bash
# 另开终端，查看 TensorBoard
tensorboard --logdir logs
# 浏览器打开 http://localhost:6006
```

---

## 评估指南

### 在验证集上评估

```bash
python scripts/eval.py --config configs/exp_imagenet_baseline.yaml --checkpoint logs/imagenet_late_fusion/best_model.pth --split val
```

### 在测试集上评估（最终结果）

```bash
python scripts/eval.py --config configs/exp_imagenet_baseline.yaml --checkpoint logs/imagenet_late_fusion/best_model.pth --split test
```

### 查看指定 checkpoint

```bash
python scripts/eval.py --config configs/exp_imagenet_baseline.yaml --checkpoint logs/imagenet_late_fusion/checkpoint_epoch50.pth
```

### 评估输出示例

```
Evaluating: 100%|████████████| 243/243 [00:15<00:00]
Results:
  OA:   0.7234
  mIoU: 0.4512
Per-class IoU:
  Background: 0.8234
  Wheat:      0.5612
  Maize:      0.4321
  ...
```

---

## 模型架构

```
输入: Sentinel-2 时间序列 (B, T, 10, 128, 128)
  │
  ├─► 共享编码器（逐帧独立编码，权重共享）
  │     ├─ Galileo 编码器（HuggingFace transformers，遥感 SSL 预训练）
  │     └─ ImageNet 编码器（SMP ResNet50，监督预训练，基线）
  │     输出每帧 4 个空间尺度的特征图 [F1, F2, F3, F4]
  │
  ├─► 时间融合（三种策略）
  │     ├─ 瓶颈融合（中期）：编码器输出后空间注意力一次性坍缩 T
  │     ├─ 晚期融合（3D 感知 DPT，主要方法）：解码器内逐层坍缩 T
  │     └─ 决策融合（软投票，消融下界）：逐帧分割后平均 logits
  │
  └─► DPT 解码器
        ReassembleBlock → FusionBlock（含全局自注意力）→ 分割头
        输出: (B, 19, 128, 128) logits
```

## 实验设计

| # | 编码器 | 训练方式 | 含义 |
|---|---|---|---|
| 1 | 随机初始化 | 全模型微调 | 下界（无预训练） |
| 2 | ImageNet 预训练 | 线性探测（冻结编码器） | ImageNet 特征质量 |
| 3 | ImageNet 预训练 | 微调 | 标准迁移学习基线 |
| 4 | Galileo 预训练 | 线性探测（冻结编码器） | SSL 特征质量 |
| 5 | **Galileo 预训练** | **微调** | **端到端 SSL 迁移（主要方法）** |

## 项目结构

```
├── configs/               # YAML 实验配置
├── data/                  # 数据流水线
│   ├── pastis_dataset.py  # PASTIS PyTorch Dataset
│   ├── collate.py         # pad_collate：变长序列对齐
│   └── augment.py         # 空间增强（跨时序一致）
├── models/                # 模型组件
│   ├── encoders/          # Galileo / ImageNet 编码器
│   ├── fusion/            # 瓶颈融合 + 决策融合 + DPT 解码器
│   └── temporal_seg_model.py  # 完整模型组装
├── losses/losses.py       # CE / Dice / Focal / Combined
├── metrics/metrics.py     # OA、mIoU、逐类别 IoU
├── trainers/trainer.py    # AMP、梯度裁剪、Top-K checkpoint、TensorBoard
├── scripts/
│   ├── train.py           # 训练入口
│   └── eval.py            # 评估入口
└── pretrained/            # Galileo 预训练权重（运行时自动下载）
```

## 损失函数

| 配置名称 | 说明 |
|---|---|
| `ce` | 标准交叉熵（ignore_index=255） |
| `dice` | 平滑 Dice Loss，对类别不平衡鲁棒 |
| `focal` | Focal Loss（alpha=0.25, gamma=2.0） |
| `combined`（默认） | 0.5 × CE + 0.5 × Dice |

## 评估指标

- **OA**（Overall Accuracy）：总正确像素数 / 总像素数
- **mIoU**（mean IoU）：所有有效类别的 IoU 均值
- **逐类别 IoU**：每类作物单独 IoU

## 常见问题

### CUDA out of memory

减小 batch size：
```bash
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --batch-size 1
```

### 数据集找不到

确认 `data/PASTIS/metadata.geojson` 存在。可手动指定路径：
```bash
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --data-root "你的/PATH/TO/PASTIS"
```

### 训练日志在哪

默认输出到 `logs/<实验名>/`，包含：
- TensorBoard 事件文件
- `checkpoint_epoch{N}.pth`：Top-3 checkpoint
- `best_model.pth`：最优模型

### 怎么恢复训练

```bash
python scripts/train.py --config configs/exp_imagenet_baseline.yaml --resume logs/imagenet_late_fusion/checkpoint_epoch20.pth
```

## 参考资料

- [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark) — Garnot et al., ICCV 2021
- [Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models](https://github.com/Bili-Sakura/GALILEO-transformers) — Tseng et al., ICML 2025
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
