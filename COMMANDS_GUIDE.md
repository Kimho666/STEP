# Action Predictor + Diffusion Policy 操作指令手册

> 所有命令均在 `diffusion_policy` 目录下执行

---

## 一、环境准备

### 1.1 创建conda环境
```bash
conda env create -f conda_environment.yaml
conda activate robodiff
```

### 1.2 安装依赖
```bash
pip install -e .
```

---

## 二、数据准备

### 2.1 下载PushT数据集
```bash
mkdir -p data/pusht
cd data/pusht
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
cd ../..
```

---

## 三、使用官方预训练权重（推荐）

### 3.1 下载官方预训练权重
```bash
mkdir -p data/checkpoints
# 从官方release下载权重文件到 data/checkpoints/
```

### 3.2 测试加载官方权重
```bash
python load_official_weights.py --ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt --test
```

### 3.3 转换为Enhanced版本（支持初始轨迹加速）
```bash
python load_official_weights.py --ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt --convert_to_enhanced --init_steps 25 --test
```

---

## 四、训练模型

### 4.1 训练Action Predictor（必须）
```bash
python train.py --config-name=train_action_predictor_lowdim_workspace
```

#### 4.1.1 使用自定义参数训练
```bash
python train.py --config-name=train_action_predictor_lowdim_workspace \
    training.device=cuda:0 \
    training.num_epochs=3000 \
    dataloader.batch_size=256 \
    policy.model.d_model=256 \
    policy.model.n_layers=4
```

#### 4.1.2 调试模式（快速测试）
```bash
python train.py --config-name=train_action_predictor_lowdim_workspace training.debug=True
```

### 4.2 训练Diffusion Policy（可选，如已有官方权重可跳过）

#### 4.2.1 使用原版Diffusion Policy
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace
```

#### 4.2.2 使用Enhanced版本
```bash
python train.py --config-name=train_enhanced_diffusion_unet_lowdim_workspace
```

---

## 五、联合推理

### 5.1 演示模式（无需检查点，快速测试）
```bash
python demo_combined_inference.py --demo
```

### 5.2 使用训练好的检查点推理
```bash
python demo_combined_inference.py \
    --action_predictor_ckpt data/outputs/YYYY.MM.DD/HH.MM.SS_train_action_predictor_lowdim_pusht_lowdim/checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt
```

### 5.3 使用官方权重 + Action Predictor
```python
# Python代码
from load_official_weights import create_combined_policy_with_official_diffusion

combined = create_combined_policy_with_official_diffusion(
    official_diffusion_ckpt='data/checkpoints/pusht_diffusion_unet_lowdim.ckpt',
    action_predictor_ckpt='data/outputs/.../checkpoints/latest.ckpt',
    device='cuda:0',
    init_trajectory_steps=25
)

# 推理
obs_dict = {'obs': obs_tensor}  # [B, n_obs_steps, obs_dim]
result = combined.predict_action(obs_dict)
action = result['action']
```

---

## 六、评估

### 6.1 评估联合推理策略
```bash
python eval_combined_inference.py \
    --action_predictor_ckpt data/outputs/.../checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt \
    --output_dir data/outputs/eval_combined \
    --n_test 50 \
    --n_test_vis 4
```

### 6.2 评估时禁用Diffusion Refinement（仅使用Action Predictor）
```bash
python eval_combined_inference.py \
    --action_predictor_ckpt data/outputs/.../checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt \
    --output_dir data/outputs/eval_ap_only \
    --no_refinement
```

### 6.3 评估时禁用反馈机制
```bash
python eval_combined_inference.py \
    --action_predictor_ckpt data/outputs/.../checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt \
    --output_dir data/outputs/eval_no_feedback \
    --no_feedback
```

---

## 七、Python API 快速参考

### 7.1 加载官方Diffusion Policy
```python
from load_official_weights import load_official_diffusion_policy

policy, cfg = load_official_diffusion_policy(
    ckpt_path='path/to/official.ckpt',
    device='cuda:0',
    use_ema=True
)
```

### 7.2 转换为Enhanced版本
```python
from load_official_weights import load_official_as_enhanced

enhanced_policy, cfg = load_official_as_enhanced(
    ckpt_path='path/to/official.ckpt',
    device='cuda:0',
    init_trajectory_steps=25
)
```

### 7.3 创建联合推理策略（已有两个检查点）
```python
from load_official_weights import create_combined_policy_with_official_diffusion

combined = create_combined_policy_with_official_diffusion(
    official_diffusion_ckpt='path/to/diffusion.ckpt',
    action_predictor_ckpt='path/to/action_predictor.ckpt',
    device='cuda:0',
    use_ema=True,
    init_trajectory_steps=25,
    use_diffusion_refinement=True,
    feedback_to_predictor=True
)
```

### 7.4 创建联合策略（仅有官方权重，Action Predictor随机初始化）
```python
from load_official_weights import create_combined_policy_with_official_only

combined, action_predictor = create_combined_policy_with_official_only(
    official_diffusion_ckpt='path/to/official.ckpt',
    device='cuda:0',
    ap_d_model=256,
    ap_n_head=8,
    ap_n_layers=4
)
# 注意：action_predictor需要单独训练！
```

### 7.5 推理代码模板
```python
import torch

# 准备观察数据
obs = torch.randn(1, 2, 20).to('cuda:0')  # [batch, n_obs_steps, obs_dim]
obs_dict = {'obs': obs}

# 重置状态（每个episode开始时调用）
combined.reset()

# 获取动作
result = combined.predict_action(obs_dict)
action = result['action']  # [batch, n_action_steps, action_dim]

# 查看中间结果
result = combined.predict_action(obs_dict, return_intermediate=True)
init_action = result['init_action']  # Action Predictor的初始预测
final_action = result['action']       # Diffusion refinement后的最终动作
```

### 7.6 性能对比
```python
# 仅使用Action Predictor（最快，精度较低）
result = combined.predict_action_without_diffusion(obs_dict)

# 仅使用Diffusion Policy（最慢，精度最高）
result = combined.predict_action_diffusion_only(obs_dict)

# 联合推理（平衡速度和精度）
result = combined.predict_action(obs_dict)

# 查看推理统计
stats = combined.get_inference_stats()
print(f"完整Diffusion步数: {stats['diffusion_steps_without_init']}")
print(f"有初始轨迹时步数: {stats['diffusion_steps_with_init']}")
print(f"加速比: {stats['speedup_ratio']:.1f}x")
```

---

## 八、常用配置修改

### 8.1 修改Action Predictor模型大小
编辑 `diffusion_policy/config/train_action_predictor_lowdim_workspace.yaml`:
```yaml
policy:
  model:
    d_model: 512        # 增大隐藏维度
    n_head: 8           # 注意力头数
    n_layers: 6         # 增加层数
    dropout: 0.1
```

### 8.2 修改初始轨迹扩散步数
编辑 `diffusion_policy/config/train_enhanced_diffusion_unet_lowdim_workspace.yaml`:
```yaml
init_trajectory_steps: 25  # 可调整为10-50之间
```

### 8.3 修改训练参数
```yaml
training:
  num_epochs: 5000
  lr_warmup_steps: 500
  
dataloader:
  batch_size: 512  # 增大batch size
```

---

## 九、文件路径说明

| 路径 | 说明 |
|------|------|
| `data/pusht/` | PushT数据集目录 |
| `data/checkpoints/` | 预训练权重目录 |
| `data/outputs/` | 训练输出目录 |
| `data/outputs/YYYY.MM.DD/HH.MM.SS_.../checkpoints/` | 训练检查点 |
| `data/outputs/YYYY.MM.DD/HH.MM.SS_.../logs.json.txt` | 训练日志 |

---

## 十、故障排除

### 10.1 CUDA内存不足
```bash
# 减小batch size
python train.py --config-name=train_action_predictor_lowdim_workspace dataloader.batch_size=128
```

### 10.2 找不到数据集
```bash
# 检查数据路径
ls data/pusht/pusht_cchi_v7_replay.zarr
```

### 10.3 权重加载失败
```python
# 使用strict=False允许部分加载
policy.load_state_dict(state_dict, strict=False)
```

### 10.4 查看训练日志
```bash
# 查看JSON日志
cat data/outputs/.../logs.json.txt
```

---

## 十一、完整工作流示例

```bash
# 1. 进入项目目录
cd diffusion_policy

# 2. 激活环境
conda activate robodiff

# 3. 下载官方权重（假设已下载到data/checkpoints/）

# 4. 训练Action Predictor
python train.py --config-name=train_action_predictor_lowdim_workspace

# 5. 等待训练完成...（查看data/outputs/获取检查点路径）

# 6. 测试联合推理
python demo_combined_inference.py \
    --action_predictor_ckpt data/outputs/2024.XX.XX/XX.XX.XX_train_action_predictor_lowdim_pusht_lowdim/checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt

# 7. 正式评估
python eval_combined_inference.py \
    --action_predictor_ckpt data/outputs/2024.XX.XX/XX.XX.XX_train_action_predictor_lowdim_pusht_lowdim/checkpoints/latest.ckpt \
    --diffusion_policy_ckpt data/checkpoints/pusht_diffusion_unet_lowdim.ckpt \
    --output_dir data/outputs/eval_final \
    --n_test 50
```

---

*最后更新: 2025年12月17日*
