# CFM (Conditional Flow Matching) 训练与推理使用手册

本手册详细介绍如何使用 Conditional Flow Matching 策略进行机器人行为克隆任务的训练和推理。

---

## 📖 目录

1. [背景介绍](#1-背景介绍)
2. [环境准备](#2-环境准备)
3. [可用模型](#3-可用模型)
4. [训练指南](#4-训练指南)
5. [推理评估](#5-推理评估)
6. [高级配置](#6-高级配置)
7. [实验对比](#7-实验对比)
8. [常见问题](#8-常见问题)

---

## 1. 背景介绍

### 1.1 什么是 Flow Matching?

Flow Matching 是一种生成模型训练方法，通过学习从噪声分布到数据分布的**速度场**来实现生成。相比 Diffusion Model：

| 特性 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 训练目标 | 预测噪声 ε | 预测速度场 v |
| 采样方式 | SDE/ODE | 纯 ODE |
| 推理步数 | 通常 50-100 步 | 通常 5-10 步 |
| 数学形式 | 复杂 noise schedule | 简单线性插值 |

### 1.2 OT-CFM (Optimal Transport Conditional Flow Matching)

本实现采用 OT-CFM 变体，使用线性插值路径：

```
x_t = t * x_0 + (1-t) * x_1

其中:
- x_0: 噪声 (t=1 时)
- x_1: 目标数据 (t=0 时)
- v = x_0 - x_1: 速度场目标
```

### 1.3 π0.5 风格约定

本实现遵循 Physical Intelligence 的 π0/π0.5 约定：

- **时间方向**: t=1 (噪声) → t=0 (数据)
- **时间采样**: Beta(1.5, 1) 分布 × 0.999 + 0.001
- **推理步数**: 默认 10 步

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# 创建 conda 环境
conda env create -f conda_environment.yaml
conda activate diffusion_policy

# 或使用 pip
pip install -e .
```

### 2.2 数据集准备

**PushT 任务** (自动下载):
```bash
# 数据会自动下载到 data/pusht/
python train.py --config-name=train_cfm_unet_lowdim_workspace
```

**Robomimic 任务**:
```bash
# 需要手动下载数据集
# 参考: https://robomimic.github.io/docs/datasets/overview.html
```

---

## 3. 可用模型

### 3.1 CFM + U-Net (推荐入门)

- **配置文件**: `train_cfm_unet_lowdim_workspace.yaml`
- **策略类**: `CFMUnetLowdimPolicy`
- **特点**: 与原始 Diffusion Policy 架构相同，仅训练目标不同

### 3.2 CFM + Transformer (π0.5 风格)

- **配置文件**: `train_cfm_transformer_lowdim_workspace.yaml`
- **策略类**: `CFMTransformerLowdimPolicy`
- **特点**: 类似 π0.5 的 Action Expert 设计

---

## 4. 训练指南

### 4.1 基础训练命令

```bash
# CFM + U-Net (PushT 任务)
python train.py --config-name=train_cfm_unet_lowdim_workspace

# CFM + Transformer (PushT 任务)
python train.py --config-name=train_cfm_transformer_lowdim_workspace
```

### 4.2 切换任务 (Robomimic)

```bash
# Can 任务
python train.py --config-name=train_cfm_unet_lowdim_workspace task=can_lowdim

# Lift 任务
python train.py --config-name=train_cfm_unet_lowdim_workspace task=lift_lowdim

# Square 任务
python train.py --config-name=train_cfm_unet_lowdim_workspace task=square_lowdim
```

### 4.3 常用训练参数

```bash
# 指定输出文件夹
python train.py --config-name=train_cfm_unet_lowdim_workspace hydra.run.dir=outputs/my_experiment

# 更灵活的输出路径格式
python train.py --config-name=train_cfm_unet_lowdim_workspace hydra.run.dir=outputs/cfm_pusht_exp1

# 修改训练轮数
python train.py --config-name=train_cfm_unet_lowdim_workspace training.num_epochs=500

# 修改批次大小
python train.py --config-name=train_cfm_unet_lowdim_workspace dataloader.batch_size=128

# 修改学习率
python train.py --config-name=train_cfm_unet_lowdim_workspace optimizer.lr=1e-4

# 修改推理步数 (影响验证时的性能)
python train.py --config-name=train_cfm_unet_lowdim_workspace policy.num_inference_steps=20

# 修改时间采样方式
python train.py --config-name=train_cfm_unet_lowdim_workspace policy.time_sampling=uniform
# 可选: 'uniform' (均匀分布) 或 'beta' (Beta分布, π0风格)

# 指定 GPU
python train.py --config-name=train_cfm_unet_lowdim_workspace training.device=cuda:1

# 禁用 WandB
python train.py --config-name=train_cfm_unet_lowdim_workspace logging.mode=disabled
```

### 4.4 Transformer 专属参数

```bash
# 修改 Transformer 层数
python train.py --config-name=train_cfm_transformer_lowdim_workspace policy.model.n_layer=12

# 修改注意力头数
python train.py --config-name=train_cfm_transformer_lowdim_workspace policy.model.n_head=8

# 修改嵌入维度
python train.py --config-name=train_cfm_transformer_lowdim_workspace policy.model.n_emb=512
```

### 4.5 训练输出

训练完成后，检查点保存在:
```
data/outputs/<timestamp>/checkpoints/
├── latest.ckpt          # 最新检查点
├── epoch=XXX-....ckpt   # 按 epoch 保存的检查点
└── best.ckpt            # 最佳验证得分检查点 (如果配置了)
```

---

## 5. 推理评估

### 5.1 使用 CFM 专属评估脚本

```bash
# 基础评估
python eval_cfm.py -c data/outputs/<run>/checkpoints/latest.ckpt -o output/cfm_eval

# 指定推理步数
python eval_cfm.py -c checkpoints/cfm_unet.ckpt -o output/eval --steps 10

# 使用不同 ODE solver
python eval_cfm.py -c checkpoints/cfm_unet.ckpt -o output/eval --solver heun --steps 10
```

### 5.2 ODE Solver 选项

| Solver | 阶数 | NFE/步 | 精度 | 速度 |
|--------|------|--------|------|------|
| `euler` | 1 | 1 | ★★☆ | ★★★ |
| `heun` | 2 | 2 | ★★★ | ★★☆ |
| `midpoint` | 2 | 2 | ★★★ | ★★☆ |
| `rk4` | 4 | 4 | ★★★★ | ★☆☆ |

> **NFE** = Number of Function Evaluations (网络前向次数)

```bash
# Euler (默认, 最快)
python eval_cfm.py -c ckpt.ckpt -o output --solver euler --steps 10

# Heun (精度更高, 推荐)
python eval_cfm.py -c ckpt.ckpt -o output --solver heun --steps 5

# RK4 (最精确, 计算量大)
python eval_cfm.py -c ckpt.ckpt -o output --solver rk4 --steps 3
```

### 5.3 推理步数对比实验

一键对比不同推理步数的性能：

```bash
python eval_cfm.py -c checkpoints/cfm_unet.ckpt -o output/compare --compare-steps 1,3,5,10,20,50
```

输出示例:
```
==========================================================================================
推理步数对比摘要
==========================================================================================
步数     平均得分      平均推理时间(ms)    总推理时间(s)    调用次数
------------------------------------------------------------------------------------------
1        0.6831        1.234              0.987           800
5        0.8269        2.456              1.965           800
10       0.8899        4.123              3.298           800
20       0.8647        7.891              6.313           800
50       0.8592        18.234             14.587          800
==========================================================================================
```

### 5.4 不使用 EMA 模型

```bash
python eval_cfm.py -c checkpoints/cfm_unet.ckpt -o output/eval --no-ema
```

### 5.5 输出文件说明

```
output/cfm_eval/
├── eval_log.json                 # 详细评估日志 (JSON)
├── cfm_performance_summary.txt   # 性能摘要报告
├── media/                        # 评估视频
│   └── *.mp4
└── step_comparison_report.txt    # 步数对比报告 (如使用 --compare-steps)
```

---

## 6. 高级配置

### 6.1 CFM 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_inference_steps` | 10 | 推理时的 ODE 积分步数 |
| `sigma_min` | 0.0 | 最小噪声 (可设为 0.001 增加稳定性) |
| `time_sampling` | `beta` | 时间采样方式: `uniform` 或 `beta` |

### 6.2 配置文件结构

```yaml
# train_cfm_unet_lowdim_workspace.yaml

defaults:
  - _self_
  - task: pusht_lowdim  # 可覆盖为 can_lowdim, lift_lowdim 等

policy:
  _target_: diffusion_policy.policy.cfm_unet_lowdim_policy.CFMUnetLowdimPolicy
  
  # CFM 特有参数
  num_inference_steps: 10
  sigma_min: 0.0
  time_sampling: beta  # 'uniform' 或 'beta'
  
  # 网络结构
  horizon: ${task.horizon}
  n_action_steps: ${task.n_action_steps}
  n_obs_steps: ${task.n_obs_steps}
  
  model:
    _target_: diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D
    input_dim: ${task.action_dim}
    global_cond_dim: ${eval:'${task.obs_dim} * ${task.n_obs_steps}'}
    diffusion_step_embed_dim: 128
    down_dims: [256, 512, 1024]

training:
  num_epochs: 300
  use_ema: true
  ...
```

### 6.3 自定义 YAML 配置

创建自定义配置文件 `my_cfm_config.yaml`:

```yaml
defaults:
  - train_cfm_unet_lowdim_workspace
  - _self_

# 覆盖默认值
policy:
  num_inference_steps: 20
  time_sampling: uniform

training:
  num_epochs: 500
  
dataloader:
  batch_size: 256
```

使用:
```bash
python train.py --config-name=my_cfm_config
```

---

## 7. 实验对比

### 7.1 CFM vs Diffusion 对比实验

```bash
# 1. 训练 Diffusion Policy (baseline)
python train.py --config-name=train_diffusion_unet_lowdim_workspace

# 2. 训练 CFM Policy
python train.py --config-name=train_cfm_unet_lowdim_workspace

# 3. 评估 Diffusion (50 步)
python eval.py -c data/outputs/diffusion/checkpoints/latest.ckpt -o output/diffusion_50 --steps 50

# 4. 评估 CFM (10 步)
python eval_cfm.py -c data/outputs/cfm/checkpoints/latest.ckpt -o output/cfm_10 --steps 10
```

### 7.2 U-Net vs Transformer 对比

```bash
# CFM + U-Net
python train.py --config-name=train_cfm_unet_lowdim_workspace
python eval_cfm.py -c output_unet/checkpoints/latest.ckpt -o eval_unet --compare-steps 5,10,20

# CFM + Transformer
python train.py --config-name=train_cfm_transformer_lowdim_workspace
python eval_cfm.py -c output_transformer/checkpoints/latest.ckpt -o eval_transformer --compare-steps 5,10,20
```

### 7.3 预期结果 (PushT 任务)

| 方法 | 推理步数 | 平均得分 | 推理时间 |
|------|----------|----------|----------|
| Diffusion + U-Net | 100 | ~0.85 | ~50ms |
| Diffusion + U-Net (DDIM) | 10 | ~0.80 | ~5ms |
| **CFM + U-Net** | 10 | ~0.89 | ~5ms |
| **CFM + U-Net** | 5 | ~0.83 | ~2.5ms |
| **CFM + Transformer** | 10 | ~0.88 | ~6ms |

---

## 8. 常见问题

### Q1: CFM 与 Diffusion 的主要区别是什么?

**训练目标不同**:
- Diffusion: 预测添加的噪声 ε
- CFM: 预测速度场 v = x_0 - x_1

**采样方式不同**:
- Diffusion: 需要 noise scheduler (DDPM/DDIM)
- CFM: 直接 ODE 积分，无需 scheduler

### Q2: 推理步数如何选择?

推荐:
- **快速原型**: 5 步 + Heun solver
- **平衡精度速度**: 10 步 + Euler solver
- **最高精度**: 20 步 + RK4 solver

### Q3: 为什么使用 Beta 时间采样?

Beta(1.5, 1) 分布使得采样更多集中在 t 接近 1 (噪声端)，这与 π0/π0.5 的设计一致，经验上能提高训练稳定性。

### Q4: 出现 NaN 怎么办?

1. 检查学习率是否过大
2. 尝试增加 `sigma_min` (如 0.001)
3. 检查数据归一化

### Q5: 如何可视化训练曲线?

```bash
# 使用 WandB (推荐)
python train.py --config-name=train_cfm_unet_lowdim_workspace logging.mode=online

# 或使用 TensorBoard
tensorboard --logdir=data/outputs/
```

---

## 📚 参考文献

1. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al., 2023
2. [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164) - Physical Intelligence, 2024
3. [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al., 2023

---

## 🔗 快速命令参考

```bash
# ===== 训练 =====
# CFM + U-Net (PushT)
python train.py --config-name=train_cfm_unet_lowdim_workspace

# CFM + Transformer (PushT)
python train.py --config-name=train_cfm_transformer_lowdim_workspace

# CFM + U-Net (Robomimic Can)
python train.py --config-name=train_cfm_unet_lowdim_workspace task=can_lowdim
python train.py --config-name=train_cfm_transformer_lowdim_workspace task=can_lowdim

# ===== 评估 =====
# 基础评估
python eval_cfm.py -c ckpt.ckpt -o output

# 指定步数
python eval_cfm.py -c ckpt.ckpt -o output --steps 10

# 使用 Heun solver
python eval_cfm.py -c ckpt.ckpt -o output --solver heun --steps 5

# 步数对比
python eval_cfm.py -c ckpt.ckpt -o output --compare-steps 1,5,10,20,50
python eval_cfm.py -c data/outputs/cfm/lowdim/square/checkpoints/latest.ckpt -o data/outputs/cfm_compare/square --compare-steps 1,2,3,4,5,10,20,50
```
# 联合推理
python eval_cfm_combined.py \
    --ap-checkpoint weights/predictor/2layers_transformer/pusht_lowdim_latest.ckpt \
    --cfm-checkpoint data/outputs/cfm/lowdim/pusht/checkpoints/latest.ckpt \
    -o data/outputs/transformer_cfm_combined/pusht \
    --compare-steps 1,2,3,4,10


python eval_cfm_combined.py     --ap-checkpoint data/outputs/action_predictor/tool_hang_lowdim_rel/checkpoints/latest.ckpt     --cfm-checkpoint data/outputs/cfm/lowdim/tool_hang/checkpoints/latest.ckpt     -o data/outputs/transformer_cfm_combined_new/tool_hang     --compare-steps 1,2,3,4,10