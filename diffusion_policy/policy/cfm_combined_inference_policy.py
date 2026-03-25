"""
CFM Combined Inference Policy
结合 Action Predictor 和 CFM Policy 的联合推理模块

核心思路：
- VAE/Action Predictor 提供初始轨迹（接近真实数据分布）
- CFM 从初始轨迹位置（t≈0.2）开始，只需1步即可精细化到 t=0
- 这样 CFM 1步也能获得较高精度

数学原理：
- 标准 CFM: x_t = t*x_0 + (1-t)*x_1, 其中 x_0=噪声, x_1=数据
- 有初始轨迹时: x_start = start_t * noise + (1-start_t) * init_action
- 只需从 t=start_t 积分到 t=0，步数大幅减少
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import time
import numpy as np

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class CFMCombinedInferencePolicy(BaseLowdimPolicy):
    """
    CFM 联合推理策略
    
    推理流程：
    1. 使用 Action Predictor (VAE) 生成初始动作轨迹
    2. 将初始轨迹作为 CFM 的起点（在 t=start_t 位置）
    3. CFM 只需 1-2 步即可精细化到最终动作
    4. 输出反馈给下一次推理作为 prev_action 条件
    
    关键参数：
    - start_t: 初始轨迹在 flow 中的位置 (0.1-0.3 推荐)
        - start_t=0 意味着初始轨迹就是最终结果（不做 refinement）
        - start_t=0.2 意味着初始轨迹混入 20% 噪声，需要轻微 refine
        - start_t=1.0 意味着忽略初始轨迹，从纯噪声开始
    - feedback_to_predictor: 将输出反馈给 VAE 的 prev_action
    - feedback_to_cfm: 将输出反馈给 CFM 的 prev_action
    
    完整架构：
    
    prev_action ──┬──> VAE (with prev_action cond) ──> init_action
                  │                                        │
                  │                                        ↓
                  └──> CFM (with prev_action cond) ─── refinement
                                                           │
                                                           ↓
                                                      final_action
                                                           │
                                                           └──> 下一次的 prev_action
    """
    
    def __init__(
        self,
        action_predictor: BaseLowdimPolicy,  # VAE 或其他 action predictor
        cfm_policy: BaseLowdimPolicy,         # CFM policy
        # 联合推理参数
        use_cfm_refinement: bool = True,      # 是否使用 CFM refinement
        feedback_to_predictor: bool = True,   # 是否将结果反馈给 predictor
        feedback_to_cfm: bool = True,         # 是否将结果反馈给 CFM
        start_t: float = 0.2,                 # 初始轨迹在 flow 中的位置
        refinement_steps: int = 1,            # CFM refinement 步数
        noise_scale: float = 1.0,             # 添加到初始轨迹的噪声缩放
        # 维度信息（可选，会自动从 cfm_policy 获取）
        horizon: int = None,
        obs_dim: int = None,
        action_dim: int = None,
        n_action_steps: int = None,
        n_obs_steps: int = None,
    ):
        super().__init__()
        
        self.action_predictor = action_predictor
        self.cfm_policy = cfm_policy
        
        self.use_cfm_refinement = use_cfm_refinement
        self.feedback_to_predictor = feedback_to_predictor
        self.feedback_to_cfm = feedback_to_cfm
        self.start_t = start_t
        self.refinement_steps = refinement_steps
        self.noise_scale = noise_scale
        
        # 从 cfm_policy 获取维度
        self.horizon = horizon or getattr(cfm_policy, 'horizon', 16)
        self.obs_dim = obs_dim or getattr(cfm_policy, 'obs_dim', 20)
        self.action_dim = action_dim or getattr(cfm_policy, 'action_dim', 2)
        self.n_action_steps = n_action_steps or getattr(cfm_policy, 'n_action_steps', 8)
        self.n_obs_steps = n_obs_steps or getattr(cfm_policy, 'n_obs_steps', 2)
        
        # 保存上一次的输出，用于反馈
        self._last_output = None
        
        # 推理时间统计
        self._predictor_times: List[float] = []
        self._cfm_times: List[float] = []
        self._total_inference_count = 0
        
        self.normalizer = LinearNormalizer()
    
    def reset(self):
        """重置状态（新 episode 开始时调用）"""
        if hasattr(self.action_predictor, 'reset'):
            self.action_predictor.reset()
        self._last_output = None
        self._predictor_times = []
        self._cfm_times = []
        self._total_inference_count = 0
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置归一化器
        
        注意：只设置 CFM 的 normalizer，Action Predictor 保留自己的 normalizer
        这样 AP_only 模式下 Action Predictor 能正确工作
        """
        self.normalizer.load_state_dict(normalizer.state_dict())
        # 不再设置 action_predictor 的 normalizer，让它使用自己的
        # if hasattr(self.action_predictor, 'set_normalizer'):
        #     self.action_predictor.set_normalizer(normalizer)
        if hasattr(self.cfm_policy, 'set_normalizer'):
            self.cfm_policy.set_normalizer(normalizer)
    
    @property
    def device(self):
        return self.cfm_policy.device
    
    @property
    def dtype(self):
        return self.cfm_policy.dtype
    
    def _cfm_refine_from_init(
        self,
        init_action: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        condition_data: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从初始轨迹开始的 CFM refinement
        
        Args:
            init_action: 初始动作轨迹 [B, T, action_dim]（已归一化）
            cond: Transformer 条件 [B, n_obs, obs_dim]
            local_cond: U-Net 局部条件
            global_cond: U-Net 全局条件
            condition_data: 条件数据（用于 inpainting）
            condition_mask: 条件掩码
        
        Returns:
            refined_action: 精细化后的动作轨迹
        """
        device = init_action.device
        dtype = init_action.dtype
        batch_size = init_action.shape[0]
        
        # 1. 构造起始点：x_start = start_t * noise + (1-start_t) * init_action
        noise = torch.randn_like(init_action) * self.noise_scale
        x_t = self.start_t * noise + (1.0 - self.start_t) * init_action
        
        # 2. 从 t=start_t 积分到 t=0
        dt = -self.start_t / self.refinement_steps  # 负号因为 t 从 start_t 降到 0
        
        model = self.cfm_policy.model
        
        # 判断是 Transformer 还是 U-Net
        is_transformer = hasattr(self.cfm_policy, 'obs_as_cond') and self.cfm_policy.obs_as_cond
        
        for i in range(self.refinement_steps):
            t_current = self.start_t - i * (self.start_t / self.refinement_steps)
            
            # 条件掩码处理
            if condition_mask is not None and condition_data is not None:
                x_t[condition_mask] = condition_data[condition_mask]
            
            # 创建时间张量
            t_tensor = torch.full(
                (batch_size,),
                t_current * 1000,  # 缩放到 [0, 1000]
                device=device,
                dtype=torch.long
            )
            
            # 预测速度场
            if is_transformer:
                pred_v = model(sample=x_t, timestep=t_tensor, cond=cond)
            else:
                pred_v = model(x_t, t_tensor, local_cond=local_cond, global_cond=global_cond)
            
            # Euler 步进
            x_t = x_t + pred_v * dt
        
        # 最终条件应用
        if condition_mask is not None and condition_data is not None:
            x_t[condition_mask] = condition_data[condition_mask]
        
        return x_t
    
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        联合推理
        
        Args:
            obs_dict: 观测字典，必须包含 'obs' key
            return_intermediate: 是否返回中间结果
        
        Returns:
            包含 'action' 和 'action_pred' 的字典
        """
        # 1. 准备 prev_action
        prev_action_for_predictor = None
        prev_action_for_cfm = None
        
        if self._last_output is not None:
            if self.feedback_to_predictor:
                prev_action_for_predictor = self._last_output
            if self.feedback_to_cfm:
                prev_action_for_cfm = self._last_output
        
        # 2. 运行 Action Predictor
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        # 检查 predictor 是否支持 prev_action 参数
        try:
            predictor_result = self.action_predictor.predict_action(
                obs_dict, prev_action=prev_action_for_predictor
            )
        except TypeError:
            predictor_result = self.action_predictor.predict_action(obs_dict)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        predictor_time = t1 - t0
        self._predictor_times.append(predictor_time)
        
        # 获取初始轨迹（归一化后的）
        init_action = predictor_result.get('action_pred', predictor_result['action'])
        
        # 3. CFM Refinement
        cfm_time = 0.0
        if self.use_cfm_refinement and self.start_t > 0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()
            
            # 准备条件
            nobs = self.cfm_policy.normalizer['obs'].normalize(obs_dict['obs'])
            B, _, Do = nobs.shape
            To = self.cfm_policy.n_obs_steps
            T = self.cfm_policy.horizon
            Da = self.cfm_policy.action_dim
            device = self.device
            dtype = self.dtype
            
            # 归一化初始轨迹
            n_init_action = self.cfm_policy.normalizer['action'].normalize(init_action)
            
            # 确保形状正确
            if n_init_action.shape[1] != T:
                # 需要填充或截断
                if n_init_action.shape[1] < T:
                    # 填充
                    pad = torch.zeros(B, T - n_init_action.shape[1], Da, device=device, dtype=dtype)
                    n_init_action = torch.cat([n_init_action, pad], dim=1)
                else:
                    # 截断
                    n_init_action = n_init_action[:, :T]
            
            # 准备条件变量
            cond = None
            local_cond = None
            global_cond = None
            condition_data = None
            condition_mask = None
            
            is_transformer = hasattr(self.cfm_policy, 'obs_as_cond') and self.cfm_policy.obs_as_cond
            use_prev_action_in_cfm = hasattr(self.cfm_policy, 'use_prev_action') and self.cfm_policy.use_prev_action
            
            if is_transformer:
                obs_cond = nobs[:, :To]
                # 如果 CFM 支持 prev_action 条件，拼接到条件中
                if use_prev_action_in_cfm and prev_action_for_cfm is not None:
                    n_prev_action = self.cfm_policy.normalizer['action'].normalize(prev_action_for_cfm)
                    prev_feat = n_prev_action.reshape(B, 1, -1).expand(B, To, -1)
                    cond = torch.cat([obs_cond, prev_feat], dim=-1)
                else:
                    cond = obs_cond
            elif hasattr(self.cfm_policy, 'obs_as_global_cond') and self.cfm_policy.obs_as_global_cond:
                obs_feat = nobs[:, :To].reshape(B, -1)
                # 如果 CFM 支持 prev_action 条件，拼接到 global_cond 中
                if use_prev_action_in_cfm and prev_action_for_cfm is not None:
                    n_prev_action = self.cfm_policy.normalizer['action'].normalize(prev_action_for_cfm)
                    prev_action_feat = n_prev_action.reshape(B, -1)
                    global_cond = torch.cat([obs_feat, prev_action_feat], dim=-1)
                else:
                    global_cond = obs_feat
            elif hasattr(self.cfm_policy, 'obs_as_local_cond') and self.cfm_policy.obs_as_local_cond:
                local_cond = torch.zeros(B, T, Do, device=device, dtype=dtype)
                local_cond[:, :To] = nobs[:, :To]
            
            # 调用 CFM refinement
            refined_action = self._cfm_refine_from_init(
                init_action=n_init_action,
                cond=cond,
                local_cond=local_cond,
                global_cond=global_cond,
                condition_data=condition_data,
                condition_mask=condition_mask,
            )
            
            # 反归一化
            final_action_pred = self.cfm_policy.normalizer['action'].unnormalize(refined_action)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()
            cfm_time = t3 - t2
            self._cfm_times.append(cfm_time)
        else:
            # 不做 refinement，直接使用 Action Predictor 的结果
            # 注意：直接用 predictor 的 action 和 action_pred，保持与原 Diffusion 联合推理一致
            final_action = predictor_result['action']
            final_action_pred = predictor_result.get('action_pred', predictor_result['action'])
            
            # 调试信息：打印形状
            print(f"[DEBUG] AP_only 模式:")
            print(f"  predictor_result['action'] shape: {final_action.shape}")
            print(f"  predictor_result['action_pred'] shape: {final_action_pred.shape}")
            print(f"  expected action_dim (from cfm): {self.action_dim}")
            print(f"  expected n_action_steps (from cfm): {self.n_action_steps}")
            
            # 直接返回，跳过后面的切片逻辑
            if self.feedback_to_predictor:
                self._last_output = final_action_pred.detach().clone()
            
            self._total_inference_count += 1
            
            result = {
                'action': final_action,
                'action_pred': final_action_pred,
            }
            
            if return_intermediate:
                result['init_action'] = init_action
                result['predictor_result'] = predictor_result
                result['predictor_time'] = predictor_time
                result['cfm_time'] = 0.0
                result['total_time'] = predictor_time
                result['start_t'] = self.start_t
                result['refinement_steps'] = self.refinement_steps
            
            return result
        
        # 4. 提取执行动作（仅 CFM refinement 模式走到这里）
        To = self.n_obs_steps
        start = To
        if hasattr(self.cfm_policy, 'oa_step_convention') and self.cfm_policy.oa_step_convention:
            start = To - 1
        end = start + self.n_action_steps
        final_action = final_action_pred[:, start:end]
        
        # 5. 保存用于反馈
        if self.feedback_to_predictor:
            self._last_output = final_action_pred.detach().clone()
        
        # 6. 统计
        self._total_inference_count += 1
        
        result = {
            'action': final_action,
            'action_pred': final_action_pred,
        }
        
        if return_intermediate:
            result['init_action'] = init_action
            result['predictor_result'] = predictor_result
            result['predictor_time'] = predictor_time
            result['cfm_time'] = cfm_time
            result['total_time'] = predictor_time + cfm_time
            result['start_t'] = self.start_t
            result['refinement_steps'] = self.refinement_steps
        
        return result
    
    def predict_action_cfm_only(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """仅使用 CFM 推理（不使用 Action Predictor）"""
        return self.cfm_policy.predict_action(obs_dict)
    
    def predict_action_predictor_only(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """仅使用 Action Predictor 推理（不使用 CFM）"""
        return self.action_predictor.predict_action(obs_dict)
    
    def get_inference_stats(self) -> Dict[str, float]:
        """获取推理统计信息"""
        avg_predictor_time = np.mean(self._predictor_times) if self._predictor_times else 0.0
        avg_cfm_time = np.mean(self._cfm_times) if self._cfm_times else 0.0
        
        return {
            'start_t': self.start_t,
            'refinement_steps': self.refinement_steps,
            'total_inference_count': self._total_inference_count,
            'avg_predictor_time': float(avg_predictor_time),
            'avg_cfm_time': float(avg_cfm_time),
            'avg_total_time': float(avg_predictor_time + avg_cfm_time),
            'total_predictor_time': float(sum(self._predictor_times)),
            'total_cfm_time': float(sum(self._cfm_times)),
        }


def create_cfm_combined_policy(
    action_predictor_checkpoint: str,
    cfm_checkpoint: str,
    start_t: float = 0.2,
    refinement_steps: int = 1,
    device: str = 'cuda:0',
) -> CFMCombinedInferencePolicy:
    """
    便捷函数：从检查点创建联合推理策略
    
    Args:
        action_predictor_checkpoint: Action Predictor 检查点路径
        cfm_checkpoint: CFM Policy 检查点路径
        start_t: 初始轨迹位置 (推荐 0.1-0.3)
        refinement_steps: CFM 精细化步数 (推荐 1-2)
        device: 计算设备
    
    Returns:
        CFMCombinedInferencePolicy 实例
    """
    import torch
    import dill
    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    
    # 加载 Action Predictor
    print(f"Loading Action Predictor from {action_predictor_checkpoint}")
    payload_ap = torch.load(open(action_predictor_checkpoint, 'rb'), pickle_module=dill)
    cfg_ap = payload_ap['cfg']
    cls_ap = hydra.utils.get_class(cfg_ap._target_)
    workspace_ap = cls_ap(cfg_ap)
    workspace_ap.load_payload(payload_ap, exclude_keys=None, include_keys=None)
    action_predictor = workspace_ap.ema_model if cfg_ap.training.use_ema else workspace_ap.model
    action_predictor.to(device)
    action_predictor.eval()
    
    # 加载 CFM Policy
    print(f"Loading CFM Policy from {cfm_checkpoint}")
    payload_cfm = torch.load(open(cfm_checkpoint, 'rb'), pickle_module=dill)
    cfg_cfm = payload_cfm['cfg']
    cls_cfm = hydra.utils.get_class(cfg_cfm._target_)
    workspace_cfm = cls_cfm(cfg_cfm)
    workspace_cfm.load_payload(payload_cfm, exclude_keys=None, include_keys=None)
    cfm_policy = workspace_cfm.ema_model if cfg_cfm.training.use_ema else workspace_cfm.model
    cfm_policy.to(device)
    cfm_policy.eval()
    
    # 创建联合策略
    combined_policy = CFMCombinedInferencePolicy(
        action_predictor=action_predictor,
        cfm_policy=cfm_policy,
        start_t=start_t,
        refinement_steps=refinement_steps,
    )
    
    # 设置 normalizer
    combined_policy.set_normalizer(cfm_policy.normalizer)
    
    print(f"Created CFMCombinedInferencePolicy with start_t={start_t}, refinement_steps={refinement_steps}")
    
    return combined_policy
