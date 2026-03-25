"""
Action Predictor Transformer Model with Cross-Attention
用于从上一个action chunk和当前观察预测动作序列
"""

from typing import Optional, Tuple
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块：action query attends to observation keys/values
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        dim_feedforward: int = None
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # 交叉注意力 (action attends to observation)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        action_emb: torch.Tensor,
        obs_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            action_emb: [B, T_action, d_model] - 动作嵌入（作为query）
            obs_emb: [B, T_obs, d_model] - 观察嵌入（作为key/value）
            action_mask: 自注意力mask
            obs_mask: 交叉注意力mask
        Returns:
            out: [B, T_action, d_model]
        """
        # 自注意力
        action_attn, _ = self.self_attn(
            action_emb, action_emb, action_emb,
            key_padding_mask=action_mask
        )
        action_emb = self.norm1(action_emb + self.dropout(action_attn))
        
        # 交叉注意力 (action attends to observation)
        cross_attn, _ = self.cross_attn(
            action_emb, obs_emb, obs_emb,
            key_padding_mask=obs_mask
        )
        action_emb = self.norm2(action_emb + self.dropout(cross_attn))
        
        # 前馈网络
        ffn_out = self.ffn(action_emb)
        action_emb = self.norm3(action_emb + ffn_out)
        
        return action_emb


class ActionPredictorTransformer(ModuleAttrMixin):
    """
    带交叉注意力的Transformer模型，用于动作预测
    
    输入：
        - prev_action: 上一个action chunk [B, T_prev, action_dim]
        - obs: 当前观察 [B, T_obs, obs_dim]
    输出：
        - pred_action: 预测的动作序列 [B, T_pred, action_dim]
    """
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        pred_horizon: int,        # 预测的动作序列长度
        n_obs_steps: int,         # 观察步数
        prev_action_horizon: int, # 上一个action chunk的长度
        d_model: int = 256,
        n_head: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = None,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.n_obs_steps = n_obs_steps
        self.prev_action_horizon = prev_action_horizon
        self.d_model = d_model
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        # 输入嵌入层
        self.prev_action_embed = nn.Linear(action_dim, d_model)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        
        # 可学习的查询token（用于生成预测的动作序列）
        self.action_query = nn.Parameter(torch.randn(1, pred_horizon, d_model))
        
        # 位置编码
        self.pos_encoder_action = SinusoidalPositionalEncoding(d_model, max_len=max(pred_horizon, prev_action_horizon) + 10)
        self.pos_encoder_obs = SinusoidalPositionalEncoding(d_model, max_len=n_obs_steps + 10)
        
        # 条件编码器：处理prev_action和obs的融合
        self.cond_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        
        # 交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                n_head=n_head,
                dropout=dropout,
                dim_feedforward=dim_feedforward
            )
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, action_dim)
        
        # 初始化
        self._init_weights()
        
        logger.info(
            "ActionPredictorTransformer - number of parameters: %e", 
            sum(p.numel() for p in self.parameters())
        )
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 初始化可学习的action query
        nn.init.normal_(self.action_query, mean=0.0, std=0.02)
    
    def forward(
        self,
        prev_action: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            prev_action: [B, T_prev, action_dim] - 上一个action chunk（专家动作）
            obs: [B, T_obs, obs_dim] - 当前观察
        
        Returns:
            pred_action: [B, pred_horizon, action_dim] - 预测的动作序列
        """
        B = prev_action.shape[0]
        device = prev_action.device
        
        # 嵌入上一个action chunk
        prev_action_emb = self.prev_action_embed(prev_action)  # [B, T_prev, d_model]
        prev_action_emb = self.pos_encoder_action(prev_action_emb)
        
        # 嵌入观察
        obs_emb = self.obs_embed(obs)  # [B, T_obs, d_model]
        obs_emb = self.pos_encoder_obs(obs_emb)
        
        # 合并prev_action和obs作为条件
        # [B, T_prev + T_obs, d_model]
        cond_emb = torch.cat([prev_action_emb, obs_emb], dim=1)
        cond_emb = self.cond_encoder(cond_emb)
        
        # 准备action query（用于生成预测的动作）
        action_query = self.action_query.expand(B, -1, -1)  # [B, pred_horizon, d_model]
        action_query = self.pos_encoder_action(action_query)
        
        # 通过交叉注意力层
        action_emb = action_query
        for layer in self.cross_attention_layers:
            action_emb = layer(action_emb, cond_emb)
        
        # 输出投影
        action_emb = self.output_norm(action_emb)
        pred_action = self.output_proj(action_emb)  # [B, pred_horizon, action_dim]
        
        return pred_action
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        获取优化器参数组，区分需要weight decay和不需要的参数
        """
        decay = set()
        no_decay = set()
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'query' in name:
                no_decay.add(name)
            else:
                decay.add(name)
        
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        return optim_groups
