"""
官方权重加载工具
用于加载官方Diffusion Policy预训练权重，并与Action Predictor系统兼容使用
"""

import os
import torch
import dill
import hydra
from omegaconf import OmegaConf
from typing import Optional, Tuple, Union

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy import EnhancedDiffusionUnetLowdimPolicy
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorLowdimPolicy
from diffusion_policy.policy.combined_inference_policy import CombinedInferencePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer


def load_official_diffusion_policy(
    ckpt_path: str,
    device: str = 'cuda:0',
    use_ema: bool = True,
) -> Tuple[DiffusionUnetLowdimPolicy, OmegaConf]:
    """
    加载官方Diffusion Policy检查点
    
    Args:
        ckpt_path: 官方检查点路径
        device: 设备
        use_ema: 是否使用EMA权重（推荐）
    
    Returns:
        policy: 加载好的策略
        cfg: 配置
    """
    print(f"Loading official checkpoint from {ckpt_path}")
    
    payload = torch.load(
        open(ckpt_path, 'rb'),
        pickle_module=dill,
        map_location=device
    )
    
    cfg = payload['cfg']
    
    # 使用原始配置实例化策略
    policy = hydra.utils.instantiate(cfg.policy)
    
    # 选择要加载的权重
    if use_ema and 'ema_model' in payload['state_dicts']:
        print("Loading EMA weights")
        policy.load_state_dict(payload['state_dicts']['ema_model'])
    elif 'model' in payload['state_dicts']:
        print("Loading model weights")
        policy.load_state_dict(payload['state_dicts']['model'])
    else:
        raise ValueError("No valid state_dict found in checkpoint")
    
    policy.eval()
    policy.to(device)
    
    print(f"Loaded policy with horizon={policy.horizon}, obs_dim={policy.obs_dim}, action_dim={policy.action_dim}")
    
    return policy, cfg


def convert_official_to_enhanced(
    official_policy: DiffusionUnetLowdimPolicy,
    init_trajectory_steps: int = 25,
) -> EnhancedDiffusionUnetLowdimPolicy:
    """
    将官方DiffusionUnetLowdimPolicy转换为EnhancedDiffusionUnetLowdimPolicy
    
    Args:
        official_policy: 官方策略
        init_trajectory_steps: 有初始轨迹时的扩散步数
    
    Returns:
        enhanced_policy: 增强版策略
    """
    # 创建Enhanced版本
    enhanced_policy = EnhancedDiffusionUnetLowdimPolicy(
        model=official_policy.model,
        noise_scheduler=official_policy.noise_scheduler,
        horizon=official_policy.horizon,
        obs_dim=official_policy.obs_dim,
        action_dim=official_policy.action_dim,
        n_action_steps=official_policy.n_action_steps,
        n_obs_steps=official_policy.n_obs_steps,
        num_inference_steps=official_policy.num_inference_steps,
        init_trajectory_steps=init_trajectory_steps,
        obs_as_local_cond=official_policy.obs_as_local_cond,
        obs_as_global_cond=official_policy.obs_as_global_cond,
        pred_action_steps_only=official_policy.pred_action_steps_only,
        oa_step_convention=official_policy.oa_step_convention,
        **official_policy.kwargs
    )
    
    # 复制normalizer
    enhanced_policy.normalizer.load_state_dict(official_policy.normalizer.state_dict())
    
    # 复制mask_generator状态（如果有）
    enhanced_policy.mask_generator = official_policy.mask_generator
    
    enhanced_policy.eval()
    
    return enhanced_policy


def load_official_as_enhanced(
    ckpt_path: str,
    device: str = 'cuda:0',
    use_ema: bool = True,
    init_trajectory_steps: int = 25,
) -> Tuple[EnhancedDiffusionUnetLowdimPolicy, OmegaConf]:
    """
    加载官方检查点并转换为Enhanced版本
    
    Args:
        ckpt_path: 官方检查点路径
        device: 设备
        use_ema: 是否使用EMA权重
        init_trajectory_steps: 有初始轨迹时的扩散步数
    
    Returns:
        enhanced_policy: Enhanced版本策略
        cfg: 原始配置
    """
    # 加载官方权重
    official_policy, cfg = load_official_diffusion_policy(
        ckpt_path, device, use_ema
    )
    
    # 转换为Enhanced版本
    enhanced_policy = convert_official_to_enhanced(
        official_policy, init_trajectory_steps
    )
    enhanced_policy.to(device)
    
    print(f"Converted to Enhanced version with init_trajectory_steps={init_trajectory_steps}")
    print(f"Speedup: {enhanced_policy.num_inference_steps}/{init_trajectory_steps} = {enhanced_policy.num_inference_steps/init_trajectory_steps:.1f}x")
    
    return enhanced_policy, cfg


def create_combined_policy_with_official_diffusion(
    official_diffusion_ckpt: str,
    action_predictor_ckpt: str,
    device: str = 'cuda:0',
    use_ema: bool = True,
    init_trajectory_steps: int = 25,
    use_diffusion_refinement: bool = True,
    feedback_to_predictor: bool = True,
) -> CombinedInferencePolicy:
    """
    使用官方Diffusion Policy权重创建联合推理策略
    
    Args:
        official_diffusion_ckpt: 官方Diffusion Policy检查点路径
        action_predictor_ckpt: Action Predictor检查点路径
        device: 设备
        use_ema: 是否使用EMA权重
        init_trajectory_steps: 有初始轨迹时的扩散步数
        use_diffusion_refinement: 是否使用diffusion refinement
        feedback_to_predictor: 是否反馈给predictor
    
    Returns:
        combined_policy: 联合推理策略
    """
    # 加载官方Diffusion Policy并转换为Enhanced版本
    enhanced_diffusion, dp_cfg = load_official_as_enhanced(
        official_diffusion_ckpt, device, use_ema, init_trajectory_steps
    )
    
    # 加载Action Predictor
    print(f"\nLoading Action Predictor from {action_predictor_ckpt}")
    ap_payload = torch.load(
        open(action_predictor_ckpt, 'rb'),
        pickle_module=dill,
        map_location=device
    )
    
    ap_cfg = ap_payload['cfg']
    action_predictor = hydra.utils.instantiate(ap_cfg.policy)
    action_predictor.load_state_dict(ap_payload['state_dicts']['model'])
    action_predictor.eval()
    action_predictor.to(device)
    
    # 创建联合策略
    combined_policy = CombinedInferencePolicy(
        action_predictor=action_predictor,
        diffusion_policy=enhanced_diffusion,
        use_diffusion_refinement=use_diffusion_refinement,
        feedback_to_predictor=feedback_to_predictor,
    )
    
    # 设置normalizer（从diffusion policy）
    combined_policy.set_normalizer(enhanced_diffusion.normalizer)
    
    print(f"\nCombined policy created successfully!")
    print(f"Inference stats: {combined_policy.get_inference_stats()}")
    
    return combined_policy


def create_combined_policy_with_official_only(
    official_diffusion_ckpt: str,
    device: str = 'cuda:0',
    use_ema: bool = True,
    init_trajectory_steps: int = 25,
    # Action Predictor参数
    ap_d_model: int = 256,
    ap_n_head: int = 8,
    ap_n_layers: int = 4,
    ap_dropout: float = 0.1,
) -> Tuple[CombinedInferencePolicy, ActionPredictorLowdimPolicy]:
    """
    仅使用官方Diffusion Policy权重创建联合推理策略
    （Action Predictor随机初始化，需要训练）
    
    Args:
        official_diffusion_ckpt: 官方Diffusion Policy检查点路径
        device: 设备
        use_ema: 是否使用EMA权重
        init_trajectory_steps: 有初始轨迹时的扩散步数
        ap_*: Action Predictor的模型参数
    
    Returns:
        combined_policy: 联合推理策略（Action Predictor未训练）
        action_predictor: Action Predictor（可单独训练）
    """
    from diffusion_policy.model.action_predictor.action_predictor_transformer import ActionPredictorTransformer
    
    # 加载官方Diffusion Policy
    enhanced_diffusion, dp_cfg = load_official_as_enhanced(
        official_diffusion_ckpt, device, use_ema, init_trajectory_steps
    )
    
    # 获取维度信息
    obs_dim = enhanced_diffusion.obs_dim
    action_dim = enhanced_diffusion.action_dim
    horizon = enhanced_diffusion.horizon
    n_obs_steps = enhanced_diffusion.n_obs_steps
    n_action_steps = enhanced_diffusion.n_action_steps
    
    print(f"\nCreating Action Predictor with dims: obs={obs_dim}, action={action_dim}, horizon={horizon}")
    
    # 创建Action Predictor
    ap_model = ActionPredictorTransformer(
        action_dim=action_dim,
        obs_dim=obs_dim,
        pred_horizon=horizon,
        n_obs_steps=n_obs_steps,
        prev_action_horizon=n_action_steps,
        d_model=ap_d_model,
        n_head=ap_n_head,
        n_layers=ap_n_layers,
        dropout=ap_dropout,
    )
    
    action_predictor = ActionPredictorLowdimPolicy(
        model=ap_model,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
    )
    action_predictor.to(device)
    
    # 设置normalizer（从diffusion policy复制）
    action_predictor.set_normalizer(enhanced_diffusion.normalizer)
    
    # 创建联合策略
    combined_policy = CombinedInferencePolicy(
        action_predictor=action_predictor,
        diffusion_policy=enhanced_diffusion,
        use_diffusion_refinement=True,
        feedback_to_predictor=True,
    )
    combined_policy.set_normalizer(enhanced_diffusion.normalizer)
    
    print(f"\nCombined policy created!")
    print(f"NOTE: Action Predictor is randomly initialized and needs training!")
    print(f"Inference stats: {combined_policy.get_inference_stats()}")
    
    return combined_policy, action_predictor


# ============ 命令行工具 ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load official Diffusion Policy weights")
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to official checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no_ema', action='store_true',
                       help='Do not use EMA weights')
    parser.add_argument('--convert_to_enhanced', action='store_true',
                       help='Convert to Enhanced version')
    parser.add_argument('--init_steps', type=int, default=25,
                       help='Init trajectory steps for Enhanced version')
    parser.add_argument('--test', action='store_true',
                       help='Run a simple test')
    args = parser.parse_args()
    
    if args.convert_to_enhanced:
        policy, cfg = load_official_as_enhanced(
            args.ckpt, args.device, not args.no_ema, args.init_steps
        )
    else:
        policy, cfg = load_official_diffusion_policy(
            args.ckpt, args.device, not args.no_ema
        )
    
    if args.test:
        print("\nRunning test inference...")
        obs = torch.randn(1, policy.n_obs_steps, policy.obs_dim).to(args.device)
        with torch.no_grad():
            result = policy.predict_action({'obs': obs})
        print(f"Output action shape: {result['action'].shape}")
        print("Test passed!")


if __name__ == "__main__":
    main()
