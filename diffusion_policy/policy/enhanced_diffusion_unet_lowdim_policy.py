"""
Enhanced Diffusion UNet Lowdim Policy
支持从初始轨迹开始推理，以减少扩散步数
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class EnhancedDiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    """
    增强版Diffusion UNet Lowdim Policy
    
    支持从初始轨迹开始的推理，可以显著减少扩散步数
    """
    
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon: int, 
            obs_dim: int, 
            action_dim: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps: int = None,
            # 从初始轨迹开始的参数
            init_trajectory_steps: int = None,  # 当有初始轨迹时使用的步数
            obs_as_local_cond: bool = False,
            obs_as_global_cond: bool = False,
            pred_action_steps_only: bool = False,
            oa_step_convention: bool = False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
            
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 当有初始轨迹时使用的步数（默认是完整步数的1/4）
        if init_trajectory_steps is None:
            init_trajectory_steps = max(num_inference_steps // 4, 10)
        self.init_trajectory_steps = init_trajectory_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            local_cond: torch.Tensor = None, 
            global_cond: torch.Tensor = None,
            init_trajectory: torch.Tensor = None,  # 初始轨迹
            start_timestep: int = None,  # 从哪个timestep开始
            generator: torch.Generator = None,
            # keyword arguments to scheduler.step
            **kwargs
            ) -> torch.Tensor:
        """
        条件采样
        
        Args:
            condition_data: 条件数据
            condition_mask: 条件mask
            local_cond: 局部条件
            global_cond: 全局条件
            init_trajectory: 初始轨迹（来自action predictor）
            start_timestep: 从哪个扩散时间步开始（用于初始轨迹）
            generator: 随机数生成器
            
        Returns:
            trajectory: 生成的轨迹
        """
        model = self.model
        scheduler = self.noise_scheduler
        
        # 检测模型类型（Transformer vs UNet）
        # Transformer模型使用 'cond' 参数，UNet模型使用 'global_cond' 和 'local_cond'
        is_transformer = hasattr(model, 'obs_as_cond')
        
        # 决定使用的推理步数和起始轨迹
        if init_trajectory is not None:
            # 有初始轨迹，从中间步骤开始
            num_inference_steps = self.init_trajectory_steps
            
            # 设置timesteps
            scheduler.set_timesteps(num_inference_steps)
            
            # 计算应该从哪个timestep开始
            # 根据scheduler.timesteps确定起始点
            if start_timestep is None:
                # 默认从较早的timestep开始（噪声较大的地方）
                # 这允许diffusion model在初始轨迹基础上进行refinement
                start_timestep = scheduler.timesteps[0].item()
            
            # 给初始轨迹添加适量的噪声，然后从该点开始去噪
            noise = torch.randn(
                size=init_trajectory.shape, 
                dtype=init_trajectory.dtype,
                device=init_trajectory.device,
                generator=generator
            )
            
            # 根据起始timestep确定噪声水平
            start_timestep_tensor = torch.tensor([start_timestep], device=init_trajectory.device)
            
            # 使用scheduler的add_noise方法添加噪声
            trajectory = scheduler.add_noise(
                init_trajectory, 
                noise, 
                start_timestep_tensor
            )
        else:
            # 没有初始轨迹，从纯噪声开始
            num_inference_steps = self.num_inference_steps
            scheduler.set_timesteps(num_inference_steps)
            
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator
            )
    
        # 去噪循环
        for t in scheduler.timesteps:
            # 1. 应用条件
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 预测模型输出
            # 根据模型类型选择正确的参数传递方式
            if is_transformer:
                # Transformer模型使用 'cond' 参数
                # global_cond 的形状是 [B, n_obs_steps * obs_dim]，需要reshape为 [B, n_obs_steps, obs_dim]
                if global_cond is not None:
                    B = global_cond.shape[0]
                    cond = global_cond.reshape(B, self.n_obs_steps, -1)
                else:
                    cond = None
                model_output = model(trajectory, t, cond=cond)
            else:
                # UNet模型使用 'local_cond' 和 'global_cond' 参数
                model_output = model(trajectory, t, 
                    local_cond=local_cond, global_cond=global_cond)

            # 3. 计算前一个时间步: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        # 最终确保条件被应用
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        init_action: torch.Tensor = None  # 来自action predictor的初始动作预测
    ) -> Dict[str, torch.Tensor]:
        """
        预测动作
        
        Args:
            obs_dict: 必须包含 "obs" key
            init_action: 来自action predictor的初始动作预测 [B, horizon, action_dim]
                        如果提供，将使用较少的扩散步数进行refinement
        
        Returns:
            result: 包含 "action" key 的字典
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        # 处理初始轨迹
        init_trajectory = None
        if init_action is not None:
            # 归一化初始动作
            init_action_normalized = self.normalizer['action'].normalize(init_action)
            
            # 检查并处理horizon不匹配的情况
            init_horizon = init_action_normalized.shape[1]
            if init_horizon != T:
                # 添加探针：打印不匹配警告
                print(f"\n[WARNING] Horizon Mismatch Detected! Init: {init_horizon}, Policy: {T}")
                
                # horizon不匹配，需要调整
                if init_horizon < T:
                    # 初始轨迹较短，需要填充（使用最后一个动作重复或零填充）
                    padding = init_action_normalized[:, -1:].expand(-1, T - init_horizon, -1)
                    init_action_normalized = torch.cat([init_action_normalized, padding], dim=1)
                else:
                    # 初始轨迹较长，需要截断
                    init_action_normalized = init_action_normalized[:, :T]
            
            if self.obs_as_local_cond or self.obs_as_global_cond:
                init_trajectory = init_action_normalized
            else:
                # 需要拼接obs
                init_trajectory = torch.zeros(B, T, Da + Do, device=device, dtype=dtype)
                init_trajectory[..., :Da] = init_action_normalized
                init_trajectory[:, :To, Da:] = nobs[:, :To]

        # 处理不同的观察条件方式
        local_cond = None
        global_cond = None
        
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # 运行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            init_trajectory=init_trajectory,
            **self.kwargs
        )
        
        # 反归一化预测
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 获取动作
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'action_pred_normalized': naction_pred
        }
        
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
            
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """计算训练损失（与原版相同）"""
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        local_cond = None
        global_cond = None
        trajectory = action
        
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps:, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # 生成impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # 采样噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        
        # 前向扩散
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # 计算loss mask
        loss_mask = ~condition_mask

        # 应用条件
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # 预测噪声
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def get_inference_steps(self, with_init_trajectory: bool = False) -> int:
        """获取推理步数"""
        if with_init_trajectory:
            return self.init_trajectory_steps
        return self.num_inference_steps

