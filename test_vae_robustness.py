"""
VAE鲁棒性自测脚本
测试后验坍塌和抗干扰能力
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def robustness_test(model, obs_sample, device='cpu'):
    """
    对VAE模型进行鲁棒性测试
    :param model: 训练好的VAEModel实例
    :param obs_sample: 单个观测样本 [1, obs_dim * obs_horizon] 或 [obs_dim * obs_horizon]
    :param device: 'cuda' or 'cpu'
    """
    model.eval()
    model.to(device)

    # 确保obs_sample维度正确 [1, dim]
    if obs_sample.ndim == 1:
        obs_sample = obs_sample.unsqueeze(0)
    obs_sample = obs_sample.to(device)

    # 获取维度信息
    latent_dim = model.net.latent_dim
    pred_horizon = model.pred_horizon
    action_dim = model.net.action_dim

    print("--- 开始VAE鲁棒性测试 ---")
    print(f"预测视界: {pred_horizon}, 动作维度: {action_dim}")

    # ==========================================
    # 测试1: 后验坍塌测试 (Posterior Collapse Test)
    # 目的：固定输入obs，改变噪声z，看输出是否变化
    # ==========================================
    print("\n[测试1] 后验坍塌测试 (多样性检查)...")

    num_samples = 10
    actions_list = []

    with torch.no_grad():
        # 采样10次不同的随机z
        for i in range(num_samples):
            z_random = torch.randn(1, latent_dim).to(device)
            decoder_input = torch.cat([obs_sample, z_random], dim=-1)
            action = model.net.decoder(decoder_input).reshape(pred_horizon, action_dim)
            actions_list.append(action.cpu().numpy())

        # 采样1次零噪声 (Mean Mode)
        z_zero = torch.zeros(1, latent_dim).to(device)
        decoder_input = torch.cat([obs_sample, z_zero], dim=-1)
        action_mean = model.net.decoder(decoder_input).reshape(pred_horizon, action_dim).cpu().numpy()

    # 计算方差：如果方差极小，说明模型忽略了z（坍塌了）
    stacked_actions = np.array(actions_list)  # [10, horizon, action_dim]
    variance = np.var(stacked_actions, axis=0).mean()

    print(f"-> 输出轨迹的平均方差: {variance:.6f}")
    if variance < 1e-5:
        print("⚠️ 警告: 方差极低！发生了后验坍塌 (Posterior Collapse)。")
        print("   原因: 模型忽略了Latent Code，退化成了确定性回归网络。")
        print("   影响: 精度高，但失去多模态生成能力。")
    else:
        print("✅ 通过: 输出具有多样性，模型有效利用了Latent Space。")

    # ==========================================
    # 测试2: 极端噪声测试 (Extreme Latent Test)
    # 目的：给z极大的值，看模型是否还能输出合理的动作
    # ==========================================
    print("\n[测试2] 极端Latent测试...")
    with torch.no_grad():
        # 放大噪声 (5倍标准差)
        z_extreme = torch.randn(1, latent_dim).to(device) * 5.0
        decoder_input = torch.cat([obs_sample, z_extreme], dim=-1)
        action_extreme = model.net.decoder(decoder_input).reshape(pred_horizon, action_dim).cpu().numpy()

    # 简单的合理性检查 (假设动作归一化在-1到1之间)
    max_val = np.max(np.abs(action_extreme))
    print(f"-> 5倍噪声下的最大动作值: {max_val:.2f}")
    if max_val > 10.0:  # 阈值取决于数据归一化
        print("⚠️ 注意: 极端噪声导致输出动作爆炸，Latent空间可能不平滑。")
    else:
        print("✅ 通过: 即使在Latent边缘，输出仍然在合理范围内。")

    # ==========================================
    # 绘图可视化
    # ==========================================
    fig, axes = plt.subplots(action_dim, 1, figsize=(8, 2 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    x_axis = np.arange(pred_horizon)

    for dim in range(action_dim):
        ax = axes[dim]
        # 画出多次采样的线 (灰色细线)
        for i in range(num_samples):
            ax.plot(x_axis, actions_list[i][:, dim], color='gray', alpha=0.3, linewidth=1)

        # 画出Mean (z=0)的线 (蓝色粗线)
        ax.plot(x_axis, action_mean[:, dim], color='blue', linewidth=2, label='Mean (z=0)')

        # 画出Extreme (z*5)的线 (红色虚线)
        ax.plot(x_axis, action_extreme[:, dim], color='red', linestyle='--', linewidth=1, label='Extreme (z*5)')

        ax.set_ylabel(f'Action Dim {dim}')
        if dim == 0:
            ax.legend()

    plt.xlabel('Time Step (Horizon)')
    plt.suptitle('VAE Robustness & Diversity Check')
    plt.tight_layout()
    plt.show()

# --- 使用示例 ---
# 假设你已经有了model和data_loader
# from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel
# model = VAEModel(...)  # 加载训练好的模型
# obs_data, _ = next(iter(dataloader))
# obs_sample = obs_data['obs'][0].flatten()  # 取第一个样本并展平
# robustness_test(model, obs_sample, device='cuda')</content>
<parameter name="filePath">d:\论文\diffusion_policytest\diffusion_policy4\test_vae_robustness.py