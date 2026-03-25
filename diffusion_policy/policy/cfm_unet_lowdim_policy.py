"""
Conditional Flow Matching (OT-CFM) Lowdim Policy

Implements Optimal Transport Conditional Flow Matching as a baseline.
Key differences from Diffusion Policy:
- No noise scheduler needed
- Predicts velocity field instead of noise
- Linear interpolation path: x_t = (1-t)*x_0 + t*x_1
- Target velocity: v = x_1 - x_0
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class CFMUnetLowdimPolicy(BaseLowdimPolicy):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) Policy
    
    Uses linear interpolation paths and predicts constant velocity.
    Inference uses Euler ODE solver.
    """
    
    def __init__(self, 
            model: ConditionalUnet1D,
            horizon: int, 
            obs_dim: int, 
            action_dim: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps: int = 10,
            # CFM specific params
            sigma_min: float = 0.0,  # Minimum noise at t=1 (data side)
            time_sampling: str = 'beta',  # 'uniform' or 'beta' (pi0-style)
            # Conditioning options
            obs_as_local_cond: bool = False,
            obs_as_global_cond: bool = False,
            pred_action_steps_only: bool = False,
            oa_step_convention: bool = False,
            # prev_action conditioning
            use_prev_action: bool = False,  # 是否使用上一个 action chunk 作为条件
            **kwargs):
        super().__init__()
        
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
            
        self.model = model
        self.use_prev_action = use_prev_action
        self._prev_action = None  # 存储上一次的 action 输出
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        
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
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        
    def reset(self):
        """重置状态（新 episode 开始时调用）"""
        self._prev_action = None
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            local_cond: Optional[torch.Tensor] = None, 
            global_cond: Optional[torch.Tensor] = None,
            num_inference_steps: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            ) -> torch.Tensor:
        """
        Sample using Euler ODE solver (π0-style).
        
        π0 convention:
        - t=1 → pure noise (x_0)
        - t=0 → clean data (x_1)
        - Flow goes from t=1 to t=0
        - ODE: dx/dt = v(x_t, t), where v = x_0 - x_1
        - Euler step: x_{t+dt} = x_t + v * dt (with dt < 0)
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
            
        batch_size = condition_data.shape[0]
        device = condition_data.device
        dtype = condition_data.dtype
        
        # 1. Initialize from standard Gaussian (noise at t=1)
        x_t = torch.randn(
            size=condition_data.shape, 
            dtype=dtype,
            device=device,
            generator=generator
        )
        
        # 2. Time step size (negative because we go from t=1 to t=0)
        dt = -1.0 / num_inference_steps
        
        # 3. Euler integration from t=1 to t=0
        for i in range(num_inference_steps):
            # Current time (continuous, in [0, 1])
            # Start at t=1, end at t=0
            t_current = 1.0 - i / num_inference_steps
            
            # Apply conditioning (inpainting)
            x_t[condition_mask] = condition_data[condition_mask]
            
            # Create time tensor
            # Scale to [0, 1000] for compatibility with sinusoidal embeddings
            t_tensor = torch.full(
                (batch_size,), 
                t_current * 1000,  # Scale for embedding compatibility
                device=device,
                dtype=torch.long
            )
            
            # Predict velocity (v = x_0 - x_1, pointing from data to noise)
            pred_v = self.model(
                x_t, 
                t_tensor, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            
            # Euler step: x_{t+dt} = x_t + v * dt
            # Since dt < 0 and v points toward noise, we move toward data
            x_t = x_t + pred_v * dt
        
        # Final conditioning enforcement
        x_t[condition_mask] = condition_data[condition_mask]
        
        return x_t
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], 
                        prev_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict action given observations.
        
        Args:
            obs_dict: must include "obs" key
            prev_action: 上一次预测的 action_pred，用于条件化 [B, T, Da]
                         如果为 None 且 use_prev_action=True，则使用内部缓存的 _prev_action
        
        Returns:
            result: must include "action" key
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

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        if self.obs_as_local_cond:
            # Condition through local feature
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
        elif self.obs_as_global_cond:
            # Condition through global feature
            # 如果有 prev_action，将其拼接到 global_cond
            obs_feat = nobs[:, :To].reshape(nobs.shape[0], -1)
            if prev_action is not None and self.use_prev_action:
                # 归一化 prev_action 并拼接
                n_prev_action = self.normalizer['action'].normalize(prev_action)
                prev_action_feat = n_prev_action.reshape(B, -1)  # [B, T*Da]
                global_cond = torch.cat([obs_feat, prev_action_feat], dim=-1)
            else:
                global_cond = obs_feat
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
        else:
            # Condition through inpainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )
        
        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

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

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        OT-CFM Training Loss (π0-style)
        
        1. Sample t ~ Beta(1.5, 1) * 0.999 + 0.001 (or Uniform[0,1])
           - Beta sampling biases toward t=1 (data), improving training
        2. Sample x_0 ~ N(0, I)  (noise)
        3. x_1 = data (action trajectory)
        4. x_t = t*x_0 + (1-t)*x_1  (π0 convention: t=0 is data, t=1 is noise)
        5. target_v = x_0 - x_1  (velocity from data to noise)
        6. loss = MSE(model(x_t, t), target_v)
        
        Note: π0 uses REVERSED convention compared to standard CFM:
        - t=0 → clean data (x_1)
        - t=1 → pure noise (x_0)
        - Inference goes from t=1 to t=0
        """
        # Normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action  # x_1 (target data)
        
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

        # Generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        batch_size = trajectory.shape[0]
        device = trajectory.device

        # ===== OT-CFM Core (π0-style) =====
        
        # 1. Sample x_0 ~ N(0, I) (noise)
        x_0 = torch.randn_like(trajectory)
        x_1 = trajectory  # Target data (clean actions)
        
        # 2. Sample time t
        # π0 uses Beta(1.5, 1) which biases toward t=1 (noise side)
        # This helps the model see more challenging intermediate states
        if self.time_sampling == 'beta':
            # Beta(1.5, 1) * 0.999 + 0.001 to avoid exact 0 or 1
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((batch_size,)) * 0.999 + 0.001
        else:  # uniform
            t = torch.rand((batch_size,), device=device)
        
        # 3. Compute x_t (π0 convention: t=0 is data, t=1 is noise)
        # x_t = t * x_0 + (1 - t) * x_1
        # Expand t for broadcasting: [B] -> [B, 1, 1]
        t_expanded = t.view(batch_size, 1, 1)
        
        # Add small noise at t=0 (data side) for stability (sigma_min)
        if self.sigma_min > 0:
            x_t = t_expanded * x_0 + (1 - t_expanded) * x_1 + self.sigma_min * torch.randn_like(x_1)
        else:
            x_t = t_expanded * x_0 + (1 - t_expanded) * x_1
        
        # 4. Target velocity: u_t = x_0 - x_1 (from data toward noise)
        # This is the vector field we want the model to learn
        target_v = x_0 - x_1
        
        # 5. Compute loss mask
        loss_mask = ~condition_mask
        
        # 6. Apply conditioning
        x_t[condition_mask] = trajectory[condition_mask]
        
        # 7. Model prediction
        # Scale t to [0, 1000] for sinusoidal embedding compatibility
        t_scaled = (t * 1000).long()
        
        pred_v = self.model(
            x_t, 
            t_scaled, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        
        # 8. MSE Loss
        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
