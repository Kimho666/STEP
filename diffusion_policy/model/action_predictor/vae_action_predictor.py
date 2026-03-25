"""
VAE-based Action Predictor model adapted from NVIDIA I2SB implementation.

This module defines a lightweight conditional VAE for low-dimensional action prediction.
The encoder produces a Gaussian posterior over latents; the decoder maps latents and
flattened observations back to a future action chunk.
"""

# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Dict, Tuple, Optional
import os

import torch
import torch.nn as nn
import torch.distributions as dist
from torch_ema import ExponentialMovingAverage


def kl_divergence_normal(mean1: torch.Tensor, mean2: torch.Tensor,
                         std1: torch.Tensor, std2: torch.Tensor) -> torch.Tensor:
    """KL divergence between two factorized Gaussians."""
    kl = ((std2 + 1e-9).log() - (std1 + 1e-9).log()
          + (std1.pow(2) + (mean2 - mean1).pow(2)) / (2 * std2.pow(2) + 1e-9)
          - 0.5)
    return kl.sum(-1).mean()


class VAEConditionalMLP(nn.Module):
    """Simple conditional VAE with MLP encoder/decoder."""

    def __init__(self, action_dim: int, pred_horizon: int, global_cond_dim: int,
                 latent_dim: int, layer: int):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.latent_dim = latent_dim
        input_dim = global_cond_dim + action_dim * pred_horizon

        hidden_dim = layer
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加 dropout 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logstd = nn.Linear(hidden_dim, latent_dim)

        decoder_input_dim = global_cond_dim + latent_dim
        self.decoder_net = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加 dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * pred_horizon),
        )

    def encoder(self, x: torch.Tensor) -> dist.Normal:
        h = self.encoder_net(x)
        mean = self.encoder_mean(h)
        logstd = self.encoder_logstd(h).clamp(min=-6.0, max=6.0)
        std = torch.exp(logstd)
        return dist.Normal(mean, std)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(x)


class VAEModel(nn.Module):
    """Minimal VAE wrapper for action prediction."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        obs_dim: int,
        obs_horizon: int,
        latent_dim: int,
        layer: int,
        use_ema: bool = True,
        pretrain: bool = False,
        ckpt_path: Optional[str] = None,
        temperature: float = 1.0  # 采样温度，控制z噪声强度，默认1.0匹配N(0,I)
    ):
        super().__init__()

        self.pred_horizon = action_horizon
        self.action_dim = action_dim

        self.net = VAEConditionalMLP(
            action_dim=action_dim,
            pred_horizon=action_horizon,
            global_cond_dim=obs_dim * obs_horizon,
            latent_dim=latent_dim,
            layer=layer
        )

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.99) if use_ema else None
        self.anneal_factor = 0.1  # 初始 KL 权重设为 0.1，避免早期坍塌
        self.prior_policy = 'gaussian'
        self.temperature = temperature  # 采样温度

        if pretrain and ckpt_path is not None:
            checkpoint = torch.load(os.path.join(ckpt_path, "model.pt"), map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            if self.ema is not None:
                self.ema.load_state_dict(checkpoint["ema"])

    def sample(self, cond: torch.Tensor, x_prior=None, diffuse_step=None) -> torch.Tensor:
        """Sample actions given condition (flattened observations)."""
        num_sample = cond.shape[0]
        latent = torch.randn((num_sample, self.net.latent_dim), device=cond.device) * self.temperature  # 应用温度缩放，增强多样性
        action_flat = self.net.decoder(torch.cat([cond, latent], dim=-1))
        return action_flat.reshape(-1, self.net.pred_horizon, self.net.action_dim)

    def get_loss(self, batch_dict: Dict[str, torch.Tensor],
                 loss_args: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        nobs = batch_dict['obs'].to(device).float().flatten(start_dim=1)
        naction = batch_dict['action'].to(device).float()

        latent_post_dist = self.net.encoder(torch.cat([nobs, naction.flatten(1)], dim=-1))
        latent_post_rsample = latent_post_dist.rsample()
        latent_post_mean = latent_post_dist.mean
        latent_post_std = latent_post_dist.stddev

        latent_prior_mean = torch.zeros_like(latent_post_mean, device=device)
        latent_prior_std = torch.ones_like(latent_post_std, device=device)

        action_rec = self.net.decoder(torch.cat([nobs, latent_post_rsample], dim=-1))
        rec_loss = torch.nn.functional.mse_loss(action_rec, naction.flatten(1)) * 1.0  # 降低重构权重
        kl_loss = self.anneal_factor * kl_divergence_normal(latent_post_mean,
                                                            latent_prior_mean,
                                                            latent_post_std,
                                                            latent_prior_std)

        self.anneal_factor = min(self.anneal_factor + 0.001, 2.0)  # 加快 KL 退火，上限提高到 2.0

        loss = rec_loss + kl_loss
        return loss, {'loss': loss, 'kl_loss': kl_loss, 'rec_loss': rec_loss}

    def log_info(self, writer, log, loss_info: Dict, optimizer, itr: int, num_itr: int):
        writer.add_scalar(itr, 'loss', loss_info['loss'].detach())
        writer.add_scalar(itr, 'kl_loss', loss_info['kl_loss'].detach())
        writer.add_scalar(itr, 'rec_loss', loss_info['rec_loss'].detach())
        log.info(
            "train_it {}/{} | lr:{} | loss:{} | kl_loss:{} | rec_loss:{}".format(
                1 + itr,
                num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.2f}".format(loss_info['loss'].item()),
                "{:+.2f}".format(loss_info['kl_loss'].item()),
                "{:+.2f}".format(loss_info['rec_loss'].item()),
            )
        )

    def save_model(self, ckpt_path: str, itr: int):
        torch.save({
            "net": self.net.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else {},
        }, os.path.join(ckpt_path, "model.pt"))
