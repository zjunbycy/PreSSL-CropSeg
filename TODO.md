# TODO

本文档记录接下来需要推进的实验、工程和性能优化任务。优先级含义：

- P0：会直接影响主实验可信度或能否稳定训练。
- P1：能明显提升效率、可复现性或实验完整度。
- P2：增强项，时间允许时做。

## 一、实验环境与代码基准

- [ ] P0 使用已转换好的 Hugging Face 格式 Galileo 权重：
  [GALILEO-transformers](https://huggingface.co/BiliSakura/GALILEO-transformers)。
- [ ] P0 确认 `models/encoders/galileo_encoder.py` 的真实模型路径全程基于 `huggingface/transformers`，不要混用原始仓库中的非 HF 推理入口。
- [ ] P0 固定并记录关键依赖版本：`torch`、`torchvision`、`transformers`、`huggingface-hub`、`accelerate`、`segmentation-models-pytorch`。
- [ ] P0 增加环境验证脚本，例如 `scripts/check_env.py`，检查 CUDA、PASTIS、Galileo 权重、输出形状。
- [ ] P1 给 README 增加一次真实环境的版本快照，避免复现实验时依赖漂移。
- [ ] P1 为 Galileo encoder 增加更严格的输入/输出 shape 单元测试。
- [ ] P1 明确 PASTIS 标签表：本地标签值为 `0..19`，训练配置使用 `num_classes=20`。

验收标准：

- `exp_late.yaml` 能加载真实 Galileo 权重并输出 `(B, 20, 128, 128)`。
- 重新 clone 后按 README 能完成一次 smoke training。

## 二、训练过程监控与优化器选择

### 监控可视化

- [ ] P0 引入 `huggingface/accelerate`，统一设备、AMP、梯度累积和多 GPU 管理。
- [ ] P1 评估是否用 Hugging Face `Trainer` 封装当前训练过程。
- [ ] P1 将 `report_to` 设置为 `"tensorboard"`，实时记录 train loss、val loss、mIoU、learning rate、GPU memory。
- [ ] P1 保留当前轻量 PyTorch trainer 作为 fallback，避免迁移期间阻塞实验。
- [ ] P1 记录每个实验的完整 config、git commit、随机种子、数据 fold。

### 优化器替换

- [ ] P0 引入 Prodigy 自动优化器，对比默认 AdamW。
- [ ] P0 Prodigy 项目地址：
  [github.com/konstmish/prodigy](https://github.com/konstmish/prodigy.git)。
- [ ] P0 推荐初始配置：`lr=1`、`weight_decay=0.1`、`decouple=True`、`slice_p=11`。
- [ ] P1 在 `configs/default.yaml` 中增加 optimizer 配置分支：`adamw` / `sgd` / `prodigy`。
- [ ] P1 新增 Prodigy 实验配置，例如 `configs/exp_late_prodigy.yaml`。
- [ ] P1 对比 AdamW 和 Prodigy 的 loss 曲线、mIoU 曲线、训练稳定性和收敛速度。

验收标准：

- TensorBoard 能同时显示 train/val 曲线。
- Prodigy 配置能完整跑完至少 1 个 epoch，并与 AdamW 形成可比日志。

## 三、自监督学习效率提升：冻结 Encoder 特征预存

在冻结 Galileo encoder、仅训练 decoder/head 的任务中，encoder 输出不会随训练变化。为避免每个 epoch 重复推理，应加入特征缓存流程。

- [ ] P0 新增 `scripts/cache_features.py`，对 train/val/test folds 逐样本执行 Galileo encoder 推理。
- [ ] P0 将每个样本的多尺度特征保存为 `.npz` 文件。
- [ ] P0 推荐保存字段：
  - `patch_id`
  - `dates`
  - `target`
  - `feat_0`
  - `feat_1`
  - `feat_2`
  - `feat_3`
  - `encoder_name`
  - `encoder_subfolder`
  - `config_hash`
- [ ] P0 新增 `CachedFeatureDataset`，训练时直接读取 `.npz` 特征，跳过 encoder。
- [ ] P1 支持缓存完整性检查：样本数、fold、feature shape、Galileo 权重版本。
- [ ] P1 支持缓存路径配置，例如 `data/cache/galileo-base-patch8/`。
- [ ] P1 比较原始训练与缓存训练的吞吐量、显存占用和最终 mIoU。
- [ ] P2 支持压缩选项：`float32`、`float16`、按尺度分文件、mmap 读取。

验收标准：

- `exp_linear_probe.yaml` 可切换到缓存特征训练。
- 缓存训练结果与非缓存训练在同一随机种子下基本一致。
- 单 epoch 训练时间显著下降。

## 四、当前主实验补全

- [ ] P0 完整跑通 ImageNet baseline：`configs/exp_imagenet_baseline.yaml`。
- [ ] P0 完整跑通 Galileo linear probe：`configs/exp_linear_probe.yaml`。
- [ ] P0 完整跑通 Galileo late fusion：`configs/exp_late.yaml`。
- [ ] P1 跑通 bottleneck fusion 和 decision fusion 对比。
- [ ] P1 为每个实验保存 `best_model.pth`、最终验证集指标、测试集指标和 TensorBoard 曲线截图。
- [ ] P1 整理实验表格：OA、mIoU、per-class IoU、训练时间、显存占用。

## 五、稳定性与质量控制

- [ ] P0 检查 Galileo + AMP 的数值稳定性；若仍出现 `nan`，默认关闭 AMP 或使用 `bf16/fp32`。
- [ ] P0 加入 loss 数值守卫：发现 `nan/inf` 时打印 batch id、patch id、target unique values。
- [ ] P1 增加 deterministic smoke test，固定 seed 后验证一次前向和一次反传。
- [ ] P1 给 `scripts/eval.py` 增加缺失 checkpoint、类别数不匹配、config 不匹配的友好报错。
- [ ] P1 检查 class imbalance，并考虑 class weight、Dice/Focal 权重调整。
- [ ] P2 增加混淆矩阵和 per-class 可视化图。

## 六、论文与汇报材料

- [ ] P1 在 `开题汇报稿.md` 中同步当前架构和实验设计。
- [ ] P1 将 README 中的 smoke test 结果与正式实验结果分开，避免误读。
- [ ] P1 整理 Galileo 论文中的关键机制，并说明本项目如何接入 HF 权重。
- [ ] P2 输出一页实验流程图：数据、encoder、late fusion decoder、loss、metrics。

## 七、后续可选增强

- [ ] P2 增加可视化脚本：随机样本的 RGB 合成图、GT mask、预测 mask。
- [ ] P2 支持 `wandb`，但默认仍使用 TensorBoard。
- [ ] P2 增加多 GPU 训练配置。
- [ ] P2 尝试不同 Galileo 尺寸或 patch size 的权重。
- [ ] P2 评估是否需要早期融合或更强的 temporal attention baseline。
