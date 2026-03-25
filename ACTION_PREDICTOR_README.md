# Action Predictor + Diffusion Policy 两阶段推理系统

## 概述

本系统实现了一个两阶段的动作预测框架：

1. **Action Predictor**: 使用带有交叉注意力机制的Transformer模型，从上一个action chunk和当前观察快速预测初始动作序列
2. **Diffusion Policy**: 使用扩散模型对初始预测进行精细化（refinement），获得高质量的最终动作

### 优势

- **加速推理**: 从初始轨迹开始可以减少约75%的扩散步数
- **保持精度**: 通过diffusion refinement保证动作质量
- **闭环反馈**: diffusion的输出作为action predictor的条件，形成闭环

## 文件结构

```
diffusion_policy/
├── model/
│   └── action_predictor/
│       ├── __init__.py
│       └── action_predictor_transformer.py  # Action Predictor Transformer模型
├── policy/
│   ├── action_predictor_lowdim_policy.py    # Action Predictor策略
│   ├── enhanced_diffusion_unet_lowdim_policy.py  # 增强版Diffusion Policy
│   └── combined_inference_policy.py          # 联合推理策略
├── workspace/
│   └── train_action_predictor_lowdim_workspace.py  # Action Predictor训练工作空间
├── dataset/
│   └── action_predictor_dataset.py          # Action Predictor数据集
├── config/
│   ├── train_action_predictor_lowdim_workspace.yaml
│   └── train_enhanced_diffusion_unet_lowdim_workspace.yaml
├── load_official_weights.py                 # 官方权重加载工具 ⭐
├── demo_combined_inference.py               # 演示脚本
└── eval_combined_inference.py               # 评估脚本
```

## 使用方法

### 方式一：使用官方预训练权重（推荐）

如果你已有官方训练好的Diffusion Policy权重，可以直接加载使用：

```bash
# 测试加载官方权重
python load_official_weights.py --ckpt path/to/official_checkpoint.ckpt --test

# 转换为Enhanced版本（支持从初始轨迹开始）
python load_official_weights.py --ckpt path/to/official_checkpoint.ckpt --convert_to_enhanced --init_steps 25
```

#### Python API加载官方权重

```python
from load_official_weights import (
    load_official_diffusion_policy,
    load_official_as_enhanced,
    create_combined_policy_with_official_diffusion,
    create_combined_policy_with_official_only
)

# 方法1：直接加载官方权重
policy, cfg = load_official_diffusion_policy('path/to/official.ckpt')

# 方法2：加载并转换为Enhanced版本（支持初始轨迹）
enhanced_policy, cfg = load_official_as_enhanced(
    'path/to/official.ckpt',
    init_trajectory_steps=25  # 有初始轨迹时只需25步（原本100步）
)

# 方法3：使用官方权重 + 已训练的Action Predictor创建联合策略
combined = create_combined_policy_with_official_diffusion(
    official_diffusion_ckpt='path/to/official.ckpt',
    action_predictor_ckpt='path/to/action_predictor.ckpt'
)

# 方法4：使用官方权重 + 新建Action Predictor（需要训练）
combined, action_predictor = create_combined_policy_with_official_only(
    official_diffusion_ckpt='path/to/official.ckpt'
)
# 注意：此时action_predictor是随机初始化的，需要训练！
```

### 方式二：从头训练

### 1. 训练Action Predictor

```bash
# 独立训练Action Predictor
python train.py --config-name=train_action_predictor_lowdim_workspace
```

### 2. 训练Diffusion Policy

你可以使用原版或增强版的Diffusion Policy：

```bash
# 使用原版Diffusion Policy（正常训练）
python train.py --config-name=train_diffusion_unet_lowdim_workspace

# 或使用增强版（推荐）
python train.py --config-name=train_enhanced_diffusion_unet_lowdim_workspace
```

### 3. 联合推理

#### 演示模式（无需检查点）

```bash
python demo_combined_inference.py --demo
```

#### 从检查点加载

```bash
python demo_combined_inference.py \
    --action_predictor_ckpt path/to/action_predictor.ckpt \
    --diffusion_policy_ckpt path/to/diffusion_policy.ckpt
```

### 4. 评估

```bash
python eval_combined_inference.py \
    --action_predictor_ckpt path/to/action_predictor.ckpt \
    --diffusion_policy_ckpt path/to/diffusion_policy.ckpt \
    --output_dir data/outputs/eval_combined \
    --n_test 50
```

## 配置说明

### Action Predictor配置

关键参数：
- `prev_action_horizon`: 上一个action chunk的长度（默认8）
- `d_model`: Transformer隐藏维度（默认256）
- `n_head`: 注意力头数（默认8）
- `n_layers`: Transformer层数（默认4）

### Enhanced Diffusion Policy配置

关键参数：
- `num_inference_steps`: 完整推理步数（默认100）
- `init_trajectory_steps`: 有初始轨迹时的推理步数（默认25）

## 推理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        联合推理流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ 当前观察 obs │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────┐     ┌──────────────────┐             │
│  │  Action Predictor    │◄────│ prev_action      │             │
│  │  (Transformer)       │     │ (反馈/初始化)     │             │
│  └──────────┬───────────┘     └──────────────────┘             │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │  初始动作预测         │                                       │
│  │  init_action         │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │  Enhanced Diffusion  │                                       │
│  │  Policy              │                                       │
│  │  (从init开始，少量步数)│                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │  最终动作 action     │────────────┐                          │
│  └──────────────────────┘            │                          │
│                                      │ 反馈                     │
│                                      ▼                          │
│                           ┌──────────────────┐                  │
│                           │ 更新prev_action  │                  │
│                           └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## 代码示例

### Python API使用

```python
from diffusion_policy.policy.combined_inference_policy import (
    CombinedInferencePolicy,
    CombinedInferencePolicyLoader
)

# 加载联合策略
combined_policy = create_combined_policy(
    action_predictor_ckpt='path/to/ap.ckpt',
    diffusion_policy_ckpt='path/to/dp.ckpt',
    device='cuda:0',
    use_diffusion_refinement=True,
    feedback_to_predictor=True,
)

# 推理
obs_dict = {'obs': obs_tensor}  # [B, n_obs_steps, obs_dim]

# 重置状态（新episode开始）
combined_policy.reset()

# 获取动作
result = combined_policy.predict_action(obs_dict)
action = result['action']  # [B, n_action_steps, action_dim]

# 查看推理统计
stats = combined_policy.get_inference_stats()
print(f"加速比: {stats['speedup_ratio']:.2f}x")
```

### 单独使用各组件

```python
# 仅使用Action Predictor（快速但可能不够精确）
result = combined_policy.predict_action_without_diffusion(obs_dict)

# 仅使用Diffusion Policy（完整精度但较慢）
result = combined_policy.predict_action_diffusion_only(obs_dict)
```

## 实验对比

| 方法 | 推理步数 | 相对速度 | 精度 |
|------|----------|----------|------|
| Diffusion Only | 100 | 1.0x | 基准 |
| Action Predictor Only | 1 | ~100x | 较低 |
| Combined (Ours) | 25 | ~4x | 接近基准 |

## 注意事项

1. **归一化器同步**: 确保Action Predictor和Diffusion Policy使用相同的归一化器
2. **维度匹配**: 两个模型的`obs_dim`、`action_dim`、`horizon`等参数需要一致
3. **训练数据**: Action Predictor训练时需要`prev_action`，使用提供的数据集包装器
4. **内存优化**: 两个模型可以在同一GPU上运行，但需要注意显存使用

## 扩展

### 支持图像观察

如需支持图像输入，可以：
1. 创建`ActionPredictorHybridPolicy`，添加视觉编码器
2. 使用`EnhancedDiffusionUnetHybridImagePolicy`替代lowdim版本
3. 修改数据集以包含图像数据

### 调整推理速度/精度平衡

- 增加`init_trajectory_steps`：更高精度，更慢
- 减少`init_trajectory_steps`：更快，可能略有精度损失
- 调整Action Predictor的模型容量：更大模型可能提供更好的初始预测
