"""
Action Predictor Image VAE Policy
- 使用 MultiImageObsEncoder 编码图像/低维观测
- 使用 VAEModel 作为动作预测后端
- 兼容 predict_action 接口，可与 CombinedInferencePolicy 配合使用
"""
from typing import Dict, Optional
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class ActionPredictorImageVAEPolicy(BaseLowdimPolicy):
    """
    基于 CVAE 的图像动作预测策略
    
    结构：
    - encoder: MultiImageObsEncoder，将多模态图像+低维观测编码为特征向量
    - model: VAEModel，条件 VAE，输入特征向量，输出动作序列
    
    推理流程：
    1. 编码观测序列 -> (B, T, D)
    2. 展平为 (B, T*D) 作为 VAE 的 cond
    3. VAE 采样得到动作序列 (B, pred_horizon, action_dim)
    """
    
    def __init__(
        self,
        encoder: MultiImageObsEncoder,
        model: VAEModel,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # VAE 采样参数
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.temperature = temperature
        
        # 视觉编码输出维度（可能包含多相机 + low_dim）
        self.obs_feature_dim = self.encoder.output_shape()[0]
        
        # VAEModel 期望的 cond 维度 = obs_dim * obs_horizon
        # 从 VAEModel 的 net 中获取期望的 global_cond_dim
        # VAEConditionalMLP 的 input_dim = global_cond_dim + action_dim * pred_horizon
        # 所以 global_cond_dim = input_dim - action_dim * pred_horizon
        vae_input_dim = self.model.net.encoder_net[0].in_features
        vae_action_flat_dim = self.model.action_dim * self.model.pred_horizon
        self.expected_cond_dim = vae_input_dim - vae_action_flat_dim
        
        # 实际 cond 维度 = encoder 输出 * n_obs_steps
        self.actual_cond_dim = self.obs_feature_dim * self.n_obs_steps
        
        # 如果维度不匹配，添加投影层
        if self.actual_cond_dim != self.expected_cond_dim:
            print(f"[ActionPredictorImageVAEPolicy] Adding projection layer: {self.actual_cond_dim} -> {self.expected_cond_dim}")
            self.obs_projection = nn.Linear(self.actual_cond_dim, self.expected_cond_dim)
        else:
            self.obs_projection = None
        
        # 用于存储上一次的动作（用于反馈）
        self._prev_action = None
    
    @property
    def device(self):
        """获取模型所在设备"""
        # 返回 encoder 第一个参数的设备
        return next(self.encoder.parameters()).device
    
    def reset(self):
        """重置内部状态（新 episode 开始时调用）"""
        self._prev_action = None
    
    def _encode_obs_seq(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码观测序列
        
        Args:
            obs_dict: {key: (B, T, ...)} 形式的观测字典
        
        Returns:
            encoded: (B, T, D) 编码后的特征
        """
        # 获取 batch size 和时间步数
        B = None
        T = None
        flat = {}
        
        for k, v in obs_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.ndim >= 2:
                B = v.shape[0]
                T = v.shape[1] if v.ndim >= 3 else 1
            
            # 展平时间维度以便编码器处理
            if v.ndim == 5:
                # 图像: (B, T, C, H, W) -> (B*T, C, H, W)
                flat[k] = v.reshape(B * T, *v.shape[2:])
            elif v.ndim == 4:
                # 已经是 (B, C, H, W)，可能没有时间维
                flat[k] = v
            elif v.ndim == 3:
                # 低维序列: (B, T, D) -> (B*T, D)
                flat[k] = v.reshape(B * T, v.shape[2])
            elif v.ndim == 2:
                # 低维无时间: (B, D)
                flat[k] = v
            else:
                flat[k] = v
        
        # 编码
        enc = self.encoder(flat)  # (B*T, D)
        D = enc.shape[-1]
        
        # reshape 回 (B, T, D)
        if T is not None and T > 1:
            enc = enc.reshape(B, T, D)
        else:
            enc = enc.unsqueeze(1)  # (B, 1, D)
        
        return enc
    
    @torch.no_grad()
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor], 
        prev_action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        预测动作
        
        Args:
            obs_dict: 观测字典，支持两种格式：
                1) {'obs': {modality_key: (B, T, ...)}} - 原始格式
                2) {modality_key: (B, T, ...)} - runner 直接提供的格式
            prev_action: 上一次的动作（可选，用于条件生成）
        
        Returns:
            包含 'action' 和 'action_pred' 的字典
        """
        # 解包 obs_dict
        if 'obs' in obs_dict and isinstance(obs_dict['obs'], dict):
            obs_raw = obs_dict['obs']
        else:
            obs_raw = obs_dict
        
        # 编码观测
        obs_feat = self._encode_obs_seq(obs_raw)  # (B, T, D)
        B, T, D = obs_feat.shape
        
        # 截取到 n_obs_steps
        To = min(self.n_obs_steps, T)
        obs_feat = obs_feat[:, :To, :]
        
        # 展平为 (B, To * D) 作为 VAE 的 cond
        cond = obs_feat.reshape(B, -1)
        
        # 应用投影层（如果需要）
        if self.obs_projection is not None:
            cond = self.obs_projection(cond)
        
        # 设置采样温度（如果 VAEModel 支持）
        original_temp = getattr(self.model, 'temperature', None)
        if original_temp is not None:
            self.model.temperature = self.temperature
        
        # VAE 采样
        action_pred = self.model.sample(cond)  # (B, pred_horizon, action_dim)
        
        # 恢复温度
        if original_temp is not None:
            self.model.temperature = original_temp
        
        # 反归一化（如果需要）
        # 注意：VAEModel.sample 返回的是归一化空间的动作
        # 需要反归一化到原始空间
        if 'action' in self.normalizer.params_dict:
            action_pred_unnorm = self.normalizer['action'].unnormalize(action_pred)
        else:
            action_pred_unnorm = action_pred
        
        # 截取 n_action_steps
        action = action_pred_unnorm[:, :self.n_action_steps]
        
        # 保存用于反馈
        self._prev_action = action_pred.clone()
        
        return {
            'action': action,
            'action_pred': action_pred_unnorm,
            'action_pred_normalized': action_pred,
        }
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置归一化器"""
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            batch: 包含 'obs' 和 'action' 的批次数据
                - obs: {modality_key: (B, T, ...)}
                - action: (B, horizon, action_dim)
        
        Returns:
            loss: 标量损失
        """
        # 编码观测
        obs_feat = self._encode_obs_seq(batch['obs'])  # (B, T, D)
        B = obs_feat.shape[0]
        
        # 截取到 n_obs_steps 并展平
        To = min(self.n_obs_steps, obs_feat.shape[1])
        cond = obs_feat[:, :To, :].reshape(B, -1)
        
        # 应用投影层（如果需要）
        if self.obs_projection is not None:
            cond = self.obs_projection(cond)
        
        # 归一化动作
        action = batch['action']
        if 'action' in self.normalizer.params_dict:
            naction = self.normalizer['action'].normalize(action)
        else:
            naction = action
        
        # 构建 VAE 需要的 batch_dict
        # VAEModel.get_loss 期望 obs 是 (B, obs_horizon, obs_dim)，会 flatten 成 (B, cond_dim)
        # 这里 cond 已经是 (B, expected_cond_dim)，需要 reshape 回 (B, 1, expected_cond_dim) 然后 flatten
        vae_batch = {
            'obs': cond.unsqueeze(1),  # (B, 1, cond_dim)
            'action': naction,
        }
        
        # 调用 VAE 的损失函数
        loss, loss_info = self.model.get_loss(vae_batch, {}, device=self.device)
        
        return loss
    
    def get_optimizer_groups(self, weight_decay: float = 1e-3):
        """获取优化器参数组"""
        # encoder、projection 和 model 分开设置
        all_params = []
        all_params.extend(list(self.encoder.parameters()))
        if self.obs_projection is not None:
            all_params.extend(list(self.obs_projection.parameters()))
        all_params.extend(list(self.model.parameters()))
        
        return [
            {'params': all_params, 'weight_decay': weight_decay},
        ]
