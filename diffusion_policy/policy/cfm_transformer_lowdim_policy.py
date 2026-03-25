"""
CFM Transformer Lowdim Policy (π0.5-style Action Expert)

Implements Flow Matching with Transformer backbone, matching π0.5's action expert design:
- Transformer encoder-decoder architecture (vs U-Net in standard Diffusion Policy)
- Beta(1.5, 1) time sampling (π0 convention)
- Time direction: t=1 (noise) → t=0 (data)
- Velocity field prediction: v = x_0 - x_1

This serves as a fair baseline to compare:
- U-Net + Flow Matching (CFMUnetLowdimPolicy)
- Transformer + Flow Matching (this, π0.5-style)
- U-Net + Diffusion (DiffusionUnetLowdimPolicy)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class CFMTransformerLowdimPolicy(BaseLowdimPolicy):
    """
    π0.5-style Action Expert: Transformer + Conditional Flow Matching
    
    Key differences from CFMUnetLowdimPolicy:
    - Uses TransformerForDiffusion instead of ConditionalUnet1D
    - Encoder-decoder architecture with cross-attention for conditioning
    - More similar to π0.5's action expert module
    """
    
    def __init__(self, 
            model: TransformerForDiffusion,
            horizon: int, 
            obs_dim: int, 
            action_dim: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps: int = 10,
            # CFM specific params
            sigma_min: float = 0.0,
            time_sampling: str = 'beta',  # 'uniform' or 'beta' (pi0-style)
            # Conditioning options
            obs_as_cond: bool = True,
            pred_action_steps_only: bool = False,
            oa_step_convention: bool = False,
            # prev_action conditioning
            use_prev_action: bool = False,  # 是否使用上一个 action chunk 作为条件
            **kwargs):
        super().__init__()
            
        self.model = model
        self.normalizer = LinearNormalizer()
        self.use_prev_action = use_prev_action
        self._prev_action = None  # 存储上一次的 action 输出
        
        # Dimensions
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        
        # CFM params
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.time_sampling = time_sampling
        
        # Conditioning
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
    
    def reset(self):
        """重置状态（新 episode 开始时调用）"""
        self._prev_action = None
    
    # ========= inference  ============
    def conditional_sample(self, 
            cond: Optional[torch.Tensor] = None,
            num_inference_steps: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            batch_size: int = 1,
            device: torch.device = None,
            dtype: torch.dtype = None,
            ) -> torch.Tensor:
        """
        Sample using Euler ODE solver (π0-style).
        
        π0 convention:
        - t=1 → pure noise (x_0)
        - t=0 → clean data (x_1)
        - Flow goes from t=1 to t=0
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        
        # Determine output shape
        if self.pred_action_steps_only:
            shape = (batch_size, self.n_action_steps, self.action_dim)
        else:
            shape = (batch_size, self.horizon, self.action_dim)
        
        # 1. Initialize from standard Gaussian (noise at t=1)
        x_t = torch.randn(
            size=shape, 
            dtype=dtype,
            device=device,
            generator=generator
        )
        
        # 2. Time step size (negative because we go from t=1 to t=0)
        dt = -1.0 / num_inference_steps
        
        # 3. Euler integration from t=1 to t=0
        for i in range(num_inference_steps):
            # Current time (start at t=1, end at t=0)
            t_current = 1.0 - i / num_inference_steps
            
            # Create time tensor (scale to match training)
            t_tensor = torch.full(
                (batch_size,), 
                t_current * 1000,
                device=device,
                dtype=torch.long
            )
            
            # Predict velocity
            pred_v = self.model(
                sample=x_t, 
                timestep=t_tensor, 
                cond=cond
            )
            
            # Euler step: x_{t+dt} = x_t + v * dt
            x_t = x_t + pred_v * dt
        
        return x_t
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                        prev_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict action given observations.
        
        Args:
            obs_dict: must include "obs" key
            prev_action: 上一次预测的 action_pred，用于条件化 [B, T, Da]
                         如果为 None 且 use_prev_action=True，则使用内部缓存的 _prev_action
        """
        assert 'obs' in obs_dict
        
        # 处理 prev_action 条件
        if prev_action is None and self.use_prev_action:
            prev_action = self._prev_action  # 使用内部缓存
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        # Prepare conditioning
        cond = None
        if self.obs_as_cond:
            obs_cond = nobs[:, :To]  # (B, To, obs_dim)
            
            # 如果有 prev_action，将其拼接到条件中
            if prev_action is not None and self.use_prev_action:
                n_prev_action = self.normalizer['action'].normalize(prev_action)
                # 拼接 obs 和 prev_action: [B, To, obs_dim] + [B, T, action_dim] -> [B, To+T, obs_dim+action_dim]
                # 或者更简单：将 prev_action 展平后拼接到 obs_cond 的特征维度
                # 这里使用特征拼接方式：[B, To, obs_dim + T*action_dim]
                prev_feat = n_prev_action.reshape(B, 1, -1).expand(B, To, -1)  # [B, To, T*Da]
                cond = torch.cat([obs_cond, prev_feat], dim=-1)  # [B, To, obs_dim + T*Da]
            else:
                cond = obs_cond

        # Run sampling
        nsample = self.conditional_sample(
            cond=cond,
            batch_size=B,
            device=device,
            dtype=dtype,
            **self.kwargs
        )
        
        # Unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(nsample)

        # Get action for the relevant timesteps
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        # 更新内部缓存
        if self.use_prev_action:
            self._prev_action = action_pred.detach().clone()
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
            
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        π0.5-style Flow Matching Loss with Transformer
        
        1. Sample t ~ Beta(1.5, 1) * 0.999 + 0.001 (or Uniform)
        2. Sample x_0 ~ N(0, I)  (noise)
        3. x_1 = data (action trajectory)
        4. x_t = t*x_0 + (1-t)*x_1  (π0 convention)
        5. target_v = x_0 - x_1
        6. loss = MSE(model(x_t, t, cond), target_v)
        """
        # Normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # Prepare trajectory (target)
        trajectory = action  # x_1 (target data)
        
        # Prepare conditioning
        cond = None
        if self.obs_as_cond:
            cond = obs[:, :self.n_obs_steps]  # (B, To, obs_dim)
            
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]

        batch_size = trajectory.shape[0]
        device = trajectory.device

        # ===== π0.5-style Flow Matching =====
        
        # 1. Sample x_0 ~ N(0, I) (noise)
        x_0 = torch.randn_like(trajectory)
        x_1 = trajectory  # Target data (clean actions)
        
        # 2. Sample time t
        if self.time_sampling == 'beta':
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((batch_size,)) * 0.999 + 0.001
        else:  # uniform
            t = torch.rand((batch_size,), device=device)
        
        # 3. Compute x_t (π0 convention: t=0 is data, t=1 is noise)
        t_expanded = t.view(batch_size, 1, 1)
        
        if self.sigma_min > 0:
            x_t = t_expanded * x_0 + (1 - t_expanded) * x_1 + self.sigma_min * torch.randn_like(x_1)
        else:
            x_t = t_expanded * x_0 + (1 - t_expanded) * x_1
        
        # 4. Target velocity: u_t = x_0 - x_1
        target_v = x_0 - x_1
        
        # 5. Model prediction (Transformer)
        t_scaled = (t * 1000).long()
        
        pred_v = self.model(
            sample=x_t, 
            timestep=t_scaled, 
            cond=cond
        )
        
        # 6. MSE Loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss
