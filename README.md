# PreSSL-CropSeg

基于自监督预训练（Galileo, ICML 2025）的时间序列遥感**耕地语义分割**项目，使用 [PASTIS 基准数据集](https://github.com/VSainteuf/pastis-benchmark)（ICCV 2021）。ZJU-GISLAB-COURSE-2026 课程项目。

## 研究动机

核心问题：**针对耕地分割任务，遥感领域自监督预训练（Galileo）相比标准 ImageNet 预训练能带来多大提升？**

PASTIS 数据集：2,433 个 Sentinel-2 时间序列样本，128×128 像素，10 个光谱波段，每样本 38–61 个时间步，18 种作物 + 背景 = 19 类。

## 模型架构

```
输入: Sentinel-2 时间序列 (B, T, 10, 128, 128)
  │
  ├─► 共享编码器（逐帧独立编码，权重共享）
  │     ├─ Galileo 编码器（HuggingFace transformers，遥感 SSL 预训练）
  │     │     模式：per_frame（逐帧）/ joint（联合时空编码，实验性）
  │     └─ ImageNet 编码器（SMP ResNet50，ImageNet 监督预训练，消融基线）
  │     输出每帧 4 个空间尺度的特征图 [F1, F2, F3, F4]
  │
  ├─► 时间融合（三种策略，核心实验变量）
  │     ├─ 瓶颈融合（中期融合）：编码器输出后用空间注意力一次性坍缩 T 维度
  │     ├─ 晚期融合（3D 感知 DPT，主要方法）：解码器内逐层坍缩 T 维度
  │     │     坍缩模块：3D 卷积 / 注意力 CLS Token（默认注意力）
  │     └─ 决策融合（软投票，消融下界）：逐帧独立分割后平均 logits，无时间交互
  │
  └─► DPT 解码器（源自 Vision Transformers for Dense Prediction）
        ReassembleBlock → FusionBlock（含全局自注意力）→ 分割头
        输出: (B, 19, 128, 128) logits
```

## 实验设计

五组消融实验，精确隔离自监督预训练的贡献：

| # | 编码器初始化 | 训练方式 | 含义 |
|---|---|---|---|
| 1 | 随机初始化 | 全模型微调 | 下界（无预训练） |
| 2 | ImageNet 监督预训练 | 线性探测（冻结编码器） | ImageNet 特征质量 |
| 3 | ImageNet 监督预训练 | 微调 | 标准迁移学习基线 |
| 4 | Galileo SSL 预训练 | 线性探测（冻结编码器） | SSL 特征质量 |
| 5 | **Galileo SSL 预训练** | **微调** | **端到端 SSL 迁移（主要方法）** |

配置文件对应：

| 配置文件 | 编码器 | 融合策略 | 冻结编码器 | 目的 |
|---|---|---|---|---|
| `configs/default.yaml` | Galileo | 晚期融合（注意力） | 否 | 默认/主要配置 |
| `configs/exp_late.yaml` | Galileo | 晚期融合（注意力） | 否 | 与默认相同 |
| `configs/exp_bottleneck.yaml` | Galileo | 瓶颈融合 | 否 | 中期融合对比实验 |
| `configs/exp_decision.yaml` | Galileo | 决策融合 | 否 | 无时间交互消融下界 |
| `configs/exp_imagenet_baseline.yaml` | ImageNet ResNet50 | 晚期融合（注意力） | 否 | 隔离 SSL 贡献的基线 |
| `configs/exp_linear_probe.yaml` | Galileo | 晚期融合（注意力） | 是 | SSL 纯特征质量测试 |

## 项目结构

```
├── configs/              # YAML 实验配置文件
├── data/                 # 数据流水线
│   ├── pastis_dataset.py # PASTIS PyTorch Dataset（基于官方实现）
│   ├── collate.py        # pad_collate：变长序列对齐
│   └── augment.py        # 空间增强（跨时序一致翻转/旋转）
├── models/               # 模型组件
│   ├── encoders/
│   │   ├── galileo_encoder.py    # Galileo SSL 编码器（per-frame/joint）
│   │   └── imagenet_encoder.py   # ImageNet 监督编码器基线
│   ├── fusion/
│   │   ├── fusion.py             # 瓶颈融合 + 决策融合
│   │   └── temporal_dpt_decoder.py  # 时间感知 DPT 解码器（晚期融合）
│   └── temporal_seg_model.py     # 完整模型组装 + 工厂方法
├── losses/               # 损失函数
│   └── losses.py         # CE / Dice / Focal / Combined（默认 CE+Dice）
├── metrics/              # 评估指标
│   └── metrics.py        # 混淆矩阵累积、OA、mIoU、逐类别 IoU
├── trainers/             # 训练框架
│   └── trainer.py        # AMP 混合精度、梯度裁剪、Top-K checkpoint、TensorBoard
├── scripts/              # 入口脚本
│   ├── train.py          # 训练入口（支持配置覆盖）
│   └── eval.py           # 评估入口
└── 论文/                 # Galileo 论文翻译与解读
```

## 数据增强

空间增强跨所有时间帧一致应用，保证时序一致性和空间对齐：
- 水平翻转（50% 概率）
- 垂直翻转（可配置，默认关闭）
- 90° 随机旋转（k=0–3）

## 损失函数

| 配置名称 | 实现 |
|---|---|
| `ce` | 标准交叉熵（ignore_index=255） |
| `dice` | 平滑 Dice Loss，对类别不平衡鲁棒 |
| `focal` | Focal Loss（alpha=0.25, gamma=2.0） |
| `combined`（默认） | 0.5×CE + 0.5×Dice |

## 评估指标

- **OA**（总体精度）：对角线像素数 / 总像素数
- **mIoU**（平均交并比）：所有有效类别的 IoU 均值
- **逐类别 IoU**：18 种作物 + 背景各自的 IoU

## 训练超参数

- 批次大小：4
- 训练轮数：50
- 学习率：5e-4（线性探测 1e-3）
- 权重衰减：1e-4
- 优化器：AdamW
- 调度器：CosineAnnealingLR
- 梯度裁剪：1.0
- 混合精度：AMP 启用

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 PASTIS 数据集（Zenodo, 29GB）
# https://zenodo.org/record/5012942

# 训练（默认配置）
python scripts/train.py --config configs/default.yaml

# 评估
python scripts/eval.py --config configs/default.yaml --checkpoint path/to/ckpt.pth --split test
```

## 参考资料

- [PASTIS Benchmark](https://github.com/VSainteuf/pastis-benchmark) — Garnot et al., ICCV 2021
- [Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models](https://github.com/Bili-Sakura/GALILEO-transformers) — ICML 2025
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
