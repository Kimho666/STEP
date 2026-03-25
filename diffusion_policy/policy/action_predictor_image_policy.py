"""
Action Predictor Image Policy
- 将多模态图像+低维观测通过视觉编码器转为平坦特征
- 复用 ActionPredictorTransformer 进行动作序列预测
- 设计为轻量配置，适配多任务（pusht/lift/can/square/transport/tool_hang）
"""
from typing import Dict
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.action_predictor.action_predictor_transformer import ActionPredictorTransformer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class ActionPredictorImagePolicy(BaseLowdimPolicy):
    def __init__(
        self,
        encoder: MultiImageObsEncoder,
        model: ActionPredictorTransformer,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        prev_action_horizon: int = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.prev_action_horizon = prev_action_horizon if prev_action_horizon else n_action_steps
        self._prev_action = None

        # 如果视觉编码输出维度与模型期望不一致，添加线性投影对齐
        enc_out_dim = self.encoder.output_shape()[0]
        if enc_out_dim != self.model.obs_dim:
            self.obs_proj = nn.Linear(enc_out_dim, self.model.obs_dim)
        else:
            self.obs_proj = nn.Identity()

    def reset(self):
        self._prev_action = None

    def _encode_obs_seq(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs_dict: {key: [B, T, ...]} -> encoded [B, T, D]"""
        # 拉平时间维度，逐帧编码
        B = None
        T = None
        flat = {}
        for k, v in obs_dict.items():
            assert v.ndim >= 2
            B, T = v.shape[0], v.shape[1]
            flat[k] = v.reshape(B * T, *v.shape[2:])
        enc = self.encoder(flat)  # (B*T, D)
        D = enc.shape[-1]
        enc = enc.reshape(B, T, D)
        return self.obs_proj(enc)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], prev_action: torch.Tensor = None):
        assert 'obs' in obs_dict
        obs_raw = obs_dict['obs']  # dict of modalities
        # 编码观测
        obs_feat = self._encode_obs_seq(obs_raw)
        
        # 归一化观测特征（如果 normalizer 有 'obs' 键）
        # 注意：当使用 diffusion policy 的 normalizer 时，可能没有 'obs' 键
        # 此时直接使用编码后的特征
        if 'obs' in self.normalizer.params_dict:
            nobs = self.normalizer['obs'].normalize(obs_feat)
        else:
            # 没有 'obs' normalizer，直接使用编码特征
            nobs = obs_feat
        
        B, _, Do = nobs.shape
        To = self.n_obs_steps

        # 准备上一个 action chunk
        if prev_action is not None:
            nprev_action = self.normalizer['action'].normalize(prev_action)
        elif self._prev_action is not None:
            nprev_action = self._prev_action
        else:
            nprev_action = torch.zeros(
                B, self.prev_action_horizon, self.model.action_dim,
                device=self.device, dtype=self.dtype
            )
        nprev_action = nprev_action.to(device=self.device, dtype=self.dtype)

        obs_in = nobs[:, :To]
        naction_pred = self.model(nprev_action, obs_in)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action = action_pred[:, :self.n_action_steps]
        self._prev_action = naction_pred.clone()
        return {
            'action': action,
            'action_pred': action_pred,
            'action_pred_normalized': naction_pred
        }

    def predict_action_with_condition(self, obs_dict: Dict[str, torch.Tensor], condition_action: torch.Tensor):
        return self.predict_action(obs_dict, prev_action=condition_action)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # batch: {obs: {modalities}, action, prev_action}
        obs_feat = self._encode_obs_seq(batch['obs'])
        # 归一化
        nbatch = {
            'obs': self.normalizer['obs'].normalize(obs_feat),
            'action': self.normalizer['action'].normalize(batch['action']),
            'prev_action': self.normalizer['action'].normalize(batch['prev_action'])
        }
        obs_in = nbatch['obs'][:, :self.n_obs_steps]
        target = nbatch['action'][:, :self.model.pred_horizon]
        pred = self.model(nbatch['prev_action'], obs_in)
        return torch.mean((pred - target) ** 2)

    def get_optimizer_groups(self, weight_decay: float = 1e-3):
        return self.model.get_optim_groups(weight_decay)
