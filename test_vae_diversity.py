"""
测试 VAE 多样性：固定 obs，多次采样，计算动作方差
"""

import torch
import numpy as np
from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel, VAEConditionalMLP

def test_vae_diversity(ckpt_path: str, obs: torch.Tensor, num_samples: int = 100):
    """
    加载 VAE，固定 obs，多次采样，计算动作方差

    Args:
        ckpt_path: VAE checkpoint 路径
        obs: 固定观测，shape (1, obs_horizon, obs_dim)
        num_samples: 采样次数
    """
    # 加载模型（假设从 checkpoint 推断参数）
    payload = torch.load(ckpt_path, map_location='cpu')
    cfg = payload['cfg']
    # 从 cfg 提取参数（假设 cfg 有 policy.model）
    model_cfg = cfg['policy']['model']
    obs_dim = model_cfg['obs_dim']
    obs_horizon = model_cfg['obs_horizon']
    vae = VAEModel(
        action_dim=model_cfg['action_dim'],
        action_horizon=model_cfg['action_horizon'],
        obs_dim=obs_dim,
        obs_horizon=obs_horizon,
        latent_dim=model_cfg['latent_dim'],
        layer=model_cfg['layer'],
        use_ema=True
    )
    vae.load_state_dict(payload['state_dicts']['model'], strict=False)
    vae.eval()

    # 准备 obs
    obs_flat = obs.flatten(start_dim=1)  # (1, obs_dim * obs_horizon)

    actions = []
    with torch.no_grad():
        for _ in range(num_samples):
            action = vae.sample(cond=obs_flat)  # (1, action_horizon, action_dim)
            actions.append(action.squeeze(0).cpu().numpy())

    actions = np.array(actions)  # (num_samples, action_horizon, action_dim)
    variance = np.var(actions, axis=0)  # (action_horizon, action_dim)
    mean_variance = np.mean(variance)

    print(f"动作方差 (平均): {mean_variance:.6f}")
    print(f"动作方差 (每步每维): \n{variance}")

    return variance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--obs_path', type=str, help='numpy 文件路径，包含 obs')
    args = parser.parse_args()

    # 示例 obs（随机生成或从文件加载）
    if args.obs_path:
        obs = torch.from_numpy(np.load(args.obs_path)).float()
    else:
        # 从 checkpoint cfg 取 obs_dim 和 obs_horizon
        payload = torch.load(args.ckpt_path, map_location='cpu')
        cfg = payload['cfg']
        model_cfg = cfg['policy']['model']
        obs_dim = model_cfg['obs_dim']
        obs_horizon = model_cfg['obs_horizon']
        obs = torch.randn(1, obs_horizon, obs_dim)

    test_vae_diversity(args.ckpt_path, obs, num_samples=100)