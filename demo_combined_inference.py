"""
联合推理演示脚本
展示如何使用Action Predictor + Diffusion Policy进行两阶段推理

使用方法：
1. 先分别训练Action Predictor和Diffusion Policy
2. 加载两个模型的checkpoint
3. 使用CombinedInferencePolicy进行推理
"""

import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import dill

from diffusion_policy.policy.combined_inference_policy import (
    CombinedInferencePolicy,
    CombinedInferencePolicyLoader
)
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorLowdimPolicy
from diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy import EnhancedDiffusionUnetLowdimPolicy


def load_policy_from_checkpoint(ckpt_path: str, device: str = 'cuda:0'):
    """从检查点加载策略"""
    payload = torch.load(
        open(ckpt_path, 'rb'), 
        pickle_module=dill,
        map_location=device
    )
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    
    # 加载状态字典
    if 'model' in payload['state_dicts']:
        policy.load_state_dict(payload['state_dicts']['model'])
    elif 'ema_model' in payload['state_dicts']:
        policy.load_state_dict(payload['state_dicts']['ema_model'])
    
    policy.eval()
    policy.to(device)
    
    return policy, cfg


def create_combined_policy(
    action_predictor_ckpt: str,
    diffusion_policy_ckpt: str,
    device: str = 'cuda:0',
    use_diffusion_refinement: bool = True,
    feedback_to_predictor: bool = True
) -> CombinedInferencePolicy:
    """
    从检查点创建联合推理策略
    
    Args:
        action_predictor_ckpt: Action Predictor检查点路径
        diffusion_policy_ckpt: Diffusion Policy检查点路径
        device: 设备
        use_diffusion_refinement: 是否使用diffusion refinement
        feedback_to_predictor: 是否将结果反馈给predictor
    
    Returns:
        combined_policy: 联合推理策略
    """
    # 加载Action Predictor
    print(f"Loading Action Predictor from {action_predictor_ckpt}")
    action_predictor, ap_cfg = load_policy_from_checkpoint(action_predictor_ckpt, device)
    
    # 加载Diffusion Policy
    print(f"Loading Diffusion Policy from {diffusion_policy_ckpt}")
    
    # 修改配置以使用Enhanced版本
    dp_payload = torch.load(
        open(diffusion_policy_ckpt, 'rb'), 
        pickle_module=dill,
        map_location=device
    )
    dp_cfg = dp_payload['cfg']
    
    # 创建Enhanced Diffusion Policy
    # 如果原始配置是标准版本，需要转换
    if 'enhanced' not in dp_cfg.policy._target_.lower():
        # 创建一个新的Enhanced版本配置
        enhanced_cfg = OmegaConf.create({
            '_target_': 'diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy.EnhancedDiffusionUnetLowdimPolicy',
            'model': dp_cfg.policy.model,
            'noise_scheduler': dp_cfg.policy.noise_scheduler,
            'horizon': dp_cfg.policy.horizon,
            'obs_dim': dp_cfg.policy.obs_dim,
            'action_dim': dp_cfg.policy.action_dim,
            'n_action_steps': dp_cfg.policy.n_action_steps,
            'n_obs_steps': dp_cfg.policy.n_obs_steps,
            'num_inference_steps': dp_cfg.policy.get('num_inference_steps', 100),
            'init_trajectory_steps': dp_cfg.get('init_trajectory_steps', 25),
            'obs_as_local_cond': dp_cfg.policy.get('obs_as_local_cond', False),
            'obs_as_global_cond': dp_cfg.policy.get('obs_as_global_cond', True),
            'pred_action_steps_only': dp_cfg.policy.get('pred_action_steps_only', False),
            'oa_step_convention': dp_cfg.policy.get('oa_step_convention', True),
        })
        diffusion_policy = hydra.utils.instantiate(enhanced_cfg)
    else:
        diffusion_policy = hydra.utils.instantiate(dp_cfg.policy)
    
    # 加载状态字典
    if 'ema_model' in dp_payload['state_dicts']:
        diffusion_policy.load_state_dict(dp_payload['state_dicts']['ema_model'], strict=False)
    elif 'model' in dp_payload['state_dicts']:
        diffusion_policy.load_state_dict(dp_payload['state_dicts']['model'], strict=False)
    
    diffusion_policy.eval()
    diffusion_policy.to(device)
    
    # 创建联合策略
    combined_policy = CombinedInferencePolicy(
        action_predictor=action_predictor,
        diffusion_policy=diffusion_policy,
        use_diffusion_refinement=use_diffusion_refinement,
        feedback_to_predictor=feedback_to_predictor,
    )
    
    # 设置normalizer（从diffusion policy复制）
    combined_policy.set_normalizer(diffusion_policy.normalizer)
    
    print("Combined policy created successfully!")
    print(f"Inference stats: {combined_policy.get_inference_stats()}")
    
    return combined_policy


def demo_inference(combined_policy: CombinedInferencePolicy, obs: torch.Tensor):
    """
    演示推理过程
    
    Args:
        combined_policy: 联合推理策略
        obs: 观察 [B, T_obs, obs_dim]
    """
    combined_policy.reset()
    
    obs_dict = {'obs': obs}
    
    # 1. 完整的两阶段推理
    print("\n=== 两阶段推理 (Action Predictor + Diffusion Refinement) ===")
    result = combined_policy.predict_action(obs_dict, return_intermediate=True)
    print(f"初始预测 (Action Predictor): shape = {result['init_action'].shape}")
    print(f"最终动作: shape = {result['action'].shape}")
    
    # 2. 仅使用Action Predictor
    print("\n=== 仅使用Action Predictor ===")
    result_ap = combined_policy.predict_action_without_diffusion(obs_dict)
    print(f"Action Predictor输出: shape = {result_ap['action'].shape}")
    
    # 3. 仅使用Diffusion Policy（从噪声开始）
    print("\n=== 仅使用Diffusion Policy (从噪声开始) ===")
    result_dp = combined_policy.predict_action_diffusion_only(obs_dict)
    print(f"Diffusion Policy输出: shape = {result_dp['action'].shape}")
    
    # 比较推理速度
    import time
    
    n_trials = 10
    
    # 两阶段推理
    start = time.time()
    for _ in range(n_trials):
        _ = combined_policy.predict_action(obs_dict)
    combined_time = (time.time() - start) / n_trials
    
    # 仅Diffusion
    start = time.time()
    for _ in range(n_trials):
        _ = combined_policy.predict_action_diffusion_only(obs_dict)
    diffusion_only_time = (time.time() - start) / n_trials
    
    print(f"\n=== 推理时间比较 ({n_trials} trials average) ===")
    print(f"两阶段推理: {combined_time*1000:.2f} ms")
    print(f"纯Diffusion推理: {diffusion_only_time*1000:.2f} ms")
    print(f"加速比: {diffusion_only_time/combined_time:.2f}x")
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_predictor_ckpt', type=str, required=False,
                       help='Path to Action Predictor checkpoint')
    parser.add_argument('--diffusion_policy_ckpt', type=str, required=False,
                       help='Path to Diffusion Policy checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo with random input')
    args = parser.parse_args()
    
    if args.demo:
        # 演示模式：使用随机数据测试
        print("Running in demo mode with random data...")
        
        # 创建模型实例（不加载权重）
        from diffusion_policy.model.action_predictor.action_predictor_transformer import ActionPredictorTransformer
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        device = args.device
        
        # 参数
        obs_dim = 20
        action_dim = 2
        horizon = 16
        n_obs_steps = 2
        n_action_steps = 8
        
        # 创建Action Predictor
        ap_model = ActionPredictorTransformer(
            action_dim=action_dim,
            obs_dim=obs_dim,
            pred_horizon=horizon,
            n_obs_steps=n_obs_steps,
            prev_action_horizon=n_action_steps,
            d_model=256,
            n_head=8,
            n_layers=4,
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
        
        # 创建Diffusion Policy
        dp_model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * n_obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
        )
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type='epsilon',
        )
        
        diffusion_policy = EnhancedDiffusionUnetLowdimPolicy(
            model=dp_model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=100,
            init_trajectory_steps=25,
            obs_as_global_cond=True,
        )
        diffusion_policy.to(device)
        
        # 创建联合策略
        combined_policy = CombinedInferencePolicy(
            action_predictor=action_predictor,
            diffusion_policy=diffusion_policy,
        )
        
        # 创建假的normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.fit({
            'obs': torch.randn(1000, horizon, obs_dim),
            'action': torch.randn(1000, horizon, action_dim),
        }, last_n_dims=1, mode='limits')
        
        combined_policy.set_normalizer(normalizer)
        
        # 运行演示
        obs = torch.randn(1, n_obs_steps, obs_dim).to(device)
        demo_inference(combined_policy, obs)
        
    elif args.action_predictor_ckpt and args.diffusion_policy_ckpt:
        # 从检查点加载
        combined_policy = create_combined_policy(
            args.action_predictor_ckpt,
            args.diffusion_policy_ckpt,
            device=args.device,
        )
        
        # 创建测试输入
        obs = torch.randn(
            1, 
            combined_policy.n_obs_steps, 
            combined_policy.obs_dim
        ).to(args.device)
        
        demo_inference(combined_policy, obs)
        
    else:
        print("Please provide checkpoint paths or use --demo flag")
        print("Usage:")
        print("  python demo_combined_inference.py --demo")
        print("  python demo_combined_inference.py --action_predictor_ckpt <path> --diffusion_policy_ckpt <path>")


if __name__ == "__main__":
    main()
