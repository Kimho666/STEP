"""
Enhanced Diffusion UNet Image Policy
- Adds init_action refinement (start from predictor trajectory with fewer steps)
- Mirrors EnhancedDiffusionUnetLowdimPolicy logic but for image observations
"""
from typing import Dict, Optional
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class EnhancedDiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        init_trajectory_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        obs_as_local_cond: bool = False,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # kwargs passed to scheduler.step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # vision encoder
        self.obs_encoder = obs_encoder
        obs_feature_dim = self.obs_encoder.output_shape()[0]

        # build diffusion model (same as DiffusionUnetImagePolicy)
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.obs_as_local_cond = obs_as_local_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        if init_trajectory_steps is None:
            init_trajectory_steps = max(num_inference_steps // 4, 10)
        self.init_trajectory_steps = init_trajectory_steps

    # ========= inference  ============
    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image observations to features with shape [B, T, D]."""
        this_obs = dict_apply(obs_dict, lambda x: x[:, : self.n_obs_steps])
        this_obs = dict_apply(this_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        feats = self.obs_encoder(this_obs)
        B = obs_dict[next(iter(obs_dict))].shape[0]
        return feats.reshape(B, self.n_obs_steps, -1)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        init_trajectory: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler

        if init_trajectory is not None:
            num_inference_steps = self.init_trajectory_steps
            scheduler.set_timesteps(num_inference_steps)
            noise = torch.randn(
                size=init_trajectory.shape,
                dtype=init_trajectory.dtype,
                device=init_trajectory.device,
                generator=generator,
            )
            start_t = scheduler.timesteps[0].item()
            start_t_tensor = torch.tensor([start_t], device=init_trajectory.device)
            trajectory = scheduler.add_noise(init_trajectory, noise, start_t_tensor)
        else:
            num_inference_steps = self.num_inference_steps
            scheduler.set_timesteps(num_inference_steps)
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], init_action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict

        # normalize raw obs dict
        nobs_raw = self.normalizer.normalize(obs_dict['obs'])
        obs_feat = self._encode_obs(nobs_raw)
        B = obs_feat.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        # build conditioning
        global_cond = None
        if self.obs_as_global_cond:
            global_cond = obs_feat.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(size=(B, T, Da + self.obs_feature_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = obs_feat
            cond_mask[:, :To, Da:] = True

        # handle init trajectory
        init_trajectory = None
        if init_action is not None:
            init_action_normalized = self.normalizer['action'].normalize(init_action)
            init_horizon = init_action_normalized.shape[1]
            if init_horizon != T:
                if init_horizon < T:
                    padding = init_action_normalized[:, -1:].expand(-1, T - init_horizon, -1)
                    init_action_normalized = torch.cat([init_action_normalized, padding], dim=1)
                else:
                    init_action_normalized = init_action_normalized[:, :T]

            if self.obs_as_global_cond:
                init_trajectory = init_action_normalized
            else:
                init_trajectory = torch.zeros(B, T, Da + self.obs_feature_dim, device=device, dtype=dtype)
                init_trajectory[..., :Da] = init_action_normalized
                init_trajectory[:, :To, Da:] = obs_feat

        traj = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            init_trajectory=init_trajectory,
            **self.kwargs,
        )

        naction_pred = traj[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ========= training helpers =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # Keep training API same as DiffusionUnetImagePolicy
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # encode obs
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps])
        this_nobs = dict_apply(this_nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)

        # build condition
        if self.obs_as_global_cond:
            cond_data = torch.zeros_like(nactions)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            cond_data = torch.zeros(size=(batch_size, horizon, self.action_dim + self.obs_feature_dim), device=nactions.device, dtype=nactions.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, self.action_dim :] = nobs_features
            cond_mask[:, : self.n_obs_steps, self.action_dim :] = True
            global_cond = None

        # sampling
        nsample = self.conditional_sample(
            cond_data=cond_data,
            condition_mask=cond_mask,
            global_cond=global_cond,
        )
        naction_pred = nsample[..., : self.action_dim]

        # loss
        if nactions.shape[1] != horizon:
            nactions = nactions[:, :horizon]
        loss = torch.mean((naction_pred - nactions) ** 2)
        return loss

    def get_optimizer_groups(self, weight_decay: float = 1e-3):
        return self.model.get_optim_groups(weight_decay)
