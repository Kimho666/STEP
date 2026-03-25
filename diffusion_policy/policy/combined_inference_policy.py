"""
Combined Inference Policy
结合Action Predictor和Diffusion Policy的联合推理模块
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import time

from einops import rearrange

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorLowdimPolicy
from diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy import EnhancedDiffusionUnetLowdimPolicy


class CombinedInferencePolicy(BaseLowdimPolicy):
    """
    联合推理策略
    
    推理流程：
    1. 使用Action Predictor生成初始动作轨迹
    2. 将初始轨迹作为起点交给Diffusion Policy
    3. Diffusion Policy通过较少的步数快速refine得到最终动作
    4. 最终动作回传给Action Predictor作为下一次推理的条件
    
    特点：
    - 两阶段推理：粗预测 + 精细化
    - 减少扩散步数：从初始轨迹开始可减少75%的步数
    - 闭环反馈：diffusion的输出作为action predictor的条件
    """
    
    def __init__(
        self,
        action_predictor: ActionPredictorLowdimPolicy,
        diffusion_policy: EnhancedDiffusionUnetLowdimPolicy,
        # 联合推理参数
        use_diffusion_refinement: bool = True,  # 是否使用diffusion refinement
        feedback_to_predictor: bool = True,     # 是否将结果反馈给predictor
        # 维度信息
        horizon: int = None,
        obs_dim: int = None,
        action_dim: int = None,
        n_action_steps: int = None,
        n_obs_steps: int = None,
    ):
        super().__init__()
        
        self.action_predictor = action_predictor
        self.diffusion_policy = diffusion_policy
        
        self.use_diffusion_refinement = use_diffusion_refinement
        self.feedback_to_predictor = feedback_to_predictor
        
        # 从diffusion policy获取维度（如果没有显式指定）
        self.horizon = horizon or diffusion_policy.horizon
        self.obs_dim = obs_dim or diffusion_policy.obs_dim
        self.action_dim = action_dim or diffusion_policy.action_dim
        self.n_action_steps = n_action_steps or diffusion_policy.n_action_steps
        self.n_obs_steps = n_obs_steps or diffusion_policy.n_obs_steps
        
        # 保存上一次的输出，用于反馈
        self._last_diffusion_output = None
        
        # 推理时间统计
        self._predictor_times: List[float] = []  # 记录每次predictor推理时间
        self._diffusion_times: List[float] = []  # 记录每次diffusion推理时间
        self._total_inference_count = 0  # 总推理次数
        
        # normalizer会在运行时从各个policy获取
        self.normalizer = LinearNormalizer()
    
    def reset(self):
        """重置状态（新episode开始时调用）"""
        self.action_predictor.reset()
        self._last_diffusion_output = None
        # 重置时间统计
        self._predictor_times = []
        self._diffusion_times = []
        self._total_inference_count = 0
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置归一化器"""
        self.normalizer.load_state_dict(normalizer.state_dict())
        # 同步到子策略
        self.action_predictor.set_normalizer(normalizer)
        self.diffusion_policy.set_normalizer(normalizer)
    
    @property
    def device(self):
        return self.diffusion_policy.device
    
    @property
    def dtype(self):
        return self.diffusion_policy.dtype
    
    def _run_diffusion(self, obs_dict: Dict[str, torch.Tensor], init_action: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """包装 diffusion 推理，保持与 eval.py 相同的调用方式。"""
        # Try direct call first
        try:
            return self.diffusion_policy.predict_action(obs_dict, init_action=init_action)
        except Exception as e:
            # If error looks like Conv2d dimension mismatch (5D fed to conv2d),
            # try flattening only 5D image tensors (B,T,C,H,W) -> (B*T,C,H,W) and retry.
            msg = str(e)
            if any(x in msg for x in ("Expected 3D", "4D", "input of size")) or isinstance(e, RuntimeError):
                try:
                    flat_obs = {}
                    for k, v in obs_dict.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 5:
                            flat_obs[k] = rearrange(v, 'b t c h w -> (b t) c h w')
                        else:
                            flat_obs[k] = v
                    return self.diffusion_policy.predict_action(flat_obs, init_action=init_action)
                except Exception:
                    # re-raise original for debugging
                    raise e
            raise

    # --------- Helper methods for robust predictor invocation ---------
    def _collect_image_modalities(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Collect keys whose values are tensors and likely image modalities (ndim >= 3)."""
        modal = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 3:
                modal[k] = v
        return modal

    def _encode_with_encoder(self, encoder, nobs: Dict[str, torch.Tensor], n_obs_steps: int):
        """使用 encoder 编码观测，自动展平时间维度。
        
        参考 ActionPredictorImagePolicy._encode_obs_seq 的处理方式。
        """
        return encode_nobs_with_fallback(encoder, nobs, n_obs_steps)

    def _safe_call_predictor(self, obs_dict: Dict[str, torch.Tensor], prev_action: Optional[torch.Tensor]):
        """Call action_predictor.predict_action with fallbacks:
        1) original call
        2) call with {'obs': collected_modalities}
        3) if predictor has encoder, encode and call predictor.model with flattened features
        """
        try:
            return self.action_predictor.predict_action(obs_dict, prev_action=prev_action)
        except (AssertionError, KeyError, TypeError) as e:
            # try to assemble {'obs': ...}
            modal = self._collect_image_modalities(obs_dict)
            if modal:
                try:
                    return self.action_predictor.predict_action({'obs': modal}, prev_action=prev_action)
                except Exception:
                    pass

            # try to use predictor's encoder if available
            encoder = getattr(self.action_predictor, 'encoder', None) or getattr(self.action_predictor, 'obs_encoder', None)
            if encoder is not None:
                nobs = self.action_predictor.normalizer.normalize(obs_dict)
                feats = self._encode_with_encoder(encoder, nobs, getattr(self.action_predictor, 'n_obs_steps', 1))
                # if feats is (B*T, D) reshape to (B, T, D)
                if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                    B = list(modal.values())[0].shape[0] if modal else feats.shape[0] // getattr(self.action_predictor, 'n_obs_steps', 1)
                    feats = rearrange(feats, '(b t) d -> b t d', b=B)
                B = feats.shape[0]
                flat = feats.reshape(B, -1)
                try:
                    action_pred = self.action_predictor.model(flat)
                    action = self.action_predictor.normalizer['action'].unnormalize(action_pred)
                    return {'action': action, 'action_pred': action_pred}
                except Exception:
                    pass

            raise RuntimeError('ActionPredictor predict_action failed; tried fallbacks but none succeeded')

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        联合推理：
        - 若开启 refinement：先用 predictor 得到 init_action，再用 diffusion 精修（与 eval.py 相同的 predict_action 调用，仅传入 init_action）。
        - 若关闭 refinement：仅返回 predictor 结果。
        
        支持两种 obs_dict 格式：
        1) {'obs': {modality_key: tensor}} - 原始格式
        2) {modality_key: tensor} - runner 直接提供的格式
        """
        # 兼容 runner 直接提供 modality keys 的情况
        if 'obs' not in obs_dict:
            # runner 给的是 {modality_key: tensor}，包装成 {'obs': obs_dict}
            obs_dict = {'obs': obs_dict}

        # 1) 如果有反馈，取上一轮 diffusion 输出作为 predictor 条件
        prev_action = None
        if self.feedback_to_predictor and self._last_diffusion_output is not None:
            prev_action = self._last_diffusion_output

        # 2) 运行 predictor（计时可选）
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        predictor_result = self.action_predictor.predict_action(obs_dict, prev_action=prev_action)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        predictor_time = t1 - t0
        self._predictor_times.append(predictor_time)

        init_action = predictor_result['action_pred']
        diffusion_result = None
        diffusion_time = 0.0

        if self.use_diffusion_refinement:
            # 3) 用 init_action 调用 diffusion（保持与 eval.py 相同接口，只多了 init_action）
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.time()
            diffusion_result = self._run_diffusion(obs_dict, init_action=init_action)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.time()
            diffusion_time = t3 - t2
            self._diffusion_times.append(diffusion_time)

            final_action = diffusion_result['action']
            final_action_pred = diffusion_result['action_pred']

            if self.feedback_to_predictor:
                self._last_diffusion_output = final_action_pred.clone()
        else:
            # 不做 refinement，直接用 predictor 结果
            final_action = predictor_result['action']
            final_action_pred = predictor_result['action_pred']

        # 4) 计数与返回
        self._total_inference_count += 1
        result = {
            'action': final_action,
            'action_pred': final_action_pred,
        }

        if return_intermediate:
            result['init_action'] = init_action
            result['init_action_from_predictor'] = predictor_result['action']
            result['predictor_result'] = predictor_result
            if diffusion_result is not None:
                result['diffusion_result'] = diffusion_result
            result['predictor_time'] = predictor_time
            result['diffusion_time'] = diffusion_time
            result['total_time'] = predictor_time + diffusion_time

        return result
    
    def predict_action_without_diffusion(
        self, 
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        仅使用Action Predictor预测（不使用Diffusion refinement）
        用于快速推理或调试
        """
        prev_action = None
        if self.feedback_to_predictor and self._last_diffusion_output is not None:
            prev_action = self._last_diffusion_output
        
        return self.action_predictor.predict_action(obs_dict, prev_action=prev_action)
    
    def predict_action_diffusion_only(
        self, 
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        仅使用Diffusion Policy预测（不使用Action Predictor初始化）
        用于对比实验only
        """
        # 与 eval.py 中单独的 diffusion 推理保持一致：不传 init_action，不改步数/调度器
        return self._run_diffusion(obs_dict, init_action=None)

    def get_inference_stats(self) -> Dict[str, float]:
        """
        获取推理统计信息
        
        Returns:
            包含以下统计信息的字典：
            - diffusion_steps_without_init: 无初始轨迹时的扩散步数
            - diffusion_steps_with_init: 有初始轨迹时的扩散步数
            - speedup_ratio: 加速比
            - total_inference_count: 总推理次数
            - avg_predictor_time: 平均predictor推理时间（秒）
            - avg_diffusion_time: 平均diffusion推理时间（秒）
            - avg_total_time: 平均总推理时间（秒）
            - total_predictor_time: 累计predictor推理时间（秒）
            - total_diffusion_time: 累计diffusion推理时间（秒）
        """
        avg_predictor_time = sum(self._predictor_times) / len(self._predictor_times) if self._predictor_times else 0.0
        avg_diffusion_time = sum(self._diffusion_times) / len(self._diffusion_times) if self._diffusion_times else 0.0
        
        return {
            # 扩散步数统计
            'diffusion_steps_without_init': self.diffusion_policy.num_inference_steps,
            'diffusion_steps_with_init': self.diffusion_policy.init_trajectory_steps,
            'speedup_ratio': self.diffusion_policy.num_inference_steps / self.diffusion_policy.init_trajectory_steps,
            # 推理次数
            'total_inference_count': self._total_inference_count,
            # 时间统计（秒）
            'avg_predictor_time': avg_predictor_time,
            'avg_diffusion_time': avg_diffusion_time,
            'avg_total_time': avg_predictor_time + avg_diffusion_time,
            # 累计时间
            'total_predictor_time': sum(self._predictor_times),
            'total_diffusion_time': sum(self._diffusion_times),
            # 时间统计（毫秒，更直观）
            'avg_predictor_time_ms': avg_predictor_time * 1000,
            'avg_diffusion_time_ms': avg_diffusion_time * 1000,
            'avg_total_time_ms': (avg_predictor_time + avg_diffusion_time) * 1000,
        }
    
    def reset_timing_stats(self):
        """重置时间统计（不重置其他状态）"""
        self._predictor_times = []
        self._diffusion_times = []
        self._total_inference_count = 0
    
    def get_timing_summary(self) -> str:
        """获取格式化的时间统计摘要"""
        stats = self.get_inference_stats()
        return (
            f"推理时间统计 (总计 {stats['total_inference_count']} 次推理):\n"
            f"  - Action Predictor: 平均 {stats['avg_predictor_time_ms']:.2f} ms\n"
            f"  - Diffusion Policy: 平均 {stats['avg_diffusion_time_ms']:.2f} ms\n"
            f"  - 总计: 平均 {stats['avg_total_time_ms']:.2f} ms\n"
            f"  - 扩散加速比: {stats['speedup_ratio']:.1f}x "
            f"({stats['diffusion_steps_without_init']} → {stats['diffusion_steps_with_init']} steps)"
        )


def attach_robust_predictor(policy):
    """Attach a robust wrapped predict_action to an instantiated action predictor policy.

    The wrapper will try:
      1) original predict_action(obs_dict, prev_action)
      2) original predict_action({'obs': collected_modalities}, prev_action)
      3) use encoder/obs_encoder to encode, call model with flattened features
    """
    import types

    original_predict = getattr(policy, 'predict_action')

    def _collect_image_modalities(obs_dict: dict) -> dict:
        modal = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 3:
                modal[k] = v
        return modal

    def _encode_with_encoder_local(encoder, nobs: dict, n_obs_steps: int):
        # Delegate to module-level helper which handles flattening and expansion
        return encode_nobs_with_fallback(encoder, nobs, n_obs_steps)

    def wrapped_predict_action(self, obs_dict, prev_action=None):
        # 首先尝试直接调用
        try:
            return original_predict(obs_dict, prev_action)
        except (AssertionError, KeyError, TypeError):
            pass
        
        # 如果失败，尝试包装成 {'obs': modalities} 格式
        # ActionPredictorImagePolicy 期望 obs_dict = {'obs': {modality_key: (B,T,C,H,W)}}
        obs_modal = _collect_image_modalities(obs_dict)
        if obs_modal:
            try:
                return original_predict({'obs': obs_modal}, prev_action)
            except Exception:
                pass
        
        # 最后尝试手动编码并调用 model
        encoder = getattr(self, 'encoder', None) or getattr(self, 'obs_encoder', None)
        if encoder is not None:
            try:
                # 类似 ActionPredictorImagePolicy._encode_obs_seq 的处理
                # obs_modal 中的 tensor 形状: (B, T, C, H, W)
                modal_to_encode = obs_modal if obs_modal else obs_dict
                B = None
                T = None
                flat = {}
                for k, v in modal_to_encode.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        if v.ndim == 5:
                            # 图像: (B, T, C, H, W) -> (B*T, C, H, W)
                            B, T = v.shape[0], v.shape[1]
                            flat[k] = v.reshape(B * T, *v.shape[2:])
                        elif v.ndim == 4:
                            # 已经是 (B, C, H, W)
                            if B is None:
                                B = v.shape[0]
                            flat[k] = v
                        elif v.ndim == 3:
                            # (B, T, D) -> (B*T, D)
                            B, T = v.shape[0], v.shape[1]
                            flat[k] = v.reshape(B * T, v.shape[2])
                        else:
                            flat[k] = v
                    else:
                        flat[k] = v
                
                # 编码
                enc = encoder(flat)  # (B*T, D)
                D = enc.shape[-1]
                
                # 如果有 obs_proj，应用它
                obs_proj = getattr(self, 'obs_proj', None)
                if obs_proj is not None:
                    enc = obs_proj(enc)
                
                # reshape 回 (B, T, D)
                if B is not None and T is not None:
                    enc = enc.reshape(B, T, -1)
                
                # 归一化
                if hasattr(self.normalizer, 'obs') or 'obs' in self.normalizer.params_dict:
                    nobs = self.normalizer['obs'].normalize(enc)
                else:
                    nobs = enc
                
                n_obs_steps = getattr(self, 'n_obs_steps', T if T else 1)
                obs_in = nobs[:, :n_obs_steps] if nobs.ndim == 3 else nobs
                
                # 准备 prev_action
                if prev_action is not None:
                    nprev_action = self.normalizer['action'].normalize(prev_action)
                elif hasattr(self, '_prev_action') and self._prev_action is not None:
                    nprev_action = self._prev_action
                else:
                    prev_horizon = getattr(self, 'prev_action_horizon', getattr(self, 'n_action_steps', 8))
                    action_dim = getattr(self.model, 'action_dim', 7)
                    nprev_action = torch.zeros(B, prev_horizon, action_dim, device=self.device, dtype=self.dtype)
                nprev_action = nprev_action.to(device=self.device, dtype=self.dtype)
                
                # 调用 model
                naction_pred = self.model(nprev_action, obs_in)
                action_pred = self.normalizer['action'].unnormalize(naction_pred)
                n_action_steps = getattr(self, 'n_action_steps', action_pred.shape[1])
                action = action_pred[:, :n_action_steps]
                
                # 保存 prev_action
                if hasattr(self, '_prev_action'):
                    self._prev_action = naction_pred.clone()
                
                return {
                    'action': action,
                    'action_pred': action_pred,
                    'action_pred_normalized': naction_pred
                }
            except Exception:
                pass

        raise RuntimeError('ActionPredictor predict_action failed and monkey-patch fallbacks exhausted')

    policy.predict_action = types.MethodType(wrapped_predict_action, policy)
    return policy


def encode_nobs_with_fallback(encoder, nobs: Dict[str, torch.Tensor], n_obs_steps: int):
    """Module-level helper: flatten time dimension for image tensors before encoding.
    
    参考 DiffusionUnetImagePolicy 和 ActionPredictorImagePolicy 的处理方式：
    - 图像 tensor 形状: (B, T, C, H, W) -> 展平为 (B*T, C, H, W)
    - 低维 tensor 形状: (B, T, D) -> 展平为 (B*T, D)
    - encoder 返回 (B*T, D_out)
    
    Returns features (B*T, D) or (B, D) depending on encoder.
    """
    # 首先对所有 tensor 进行时间维度截取和展平（类似 dict_apply + reshape）
    B = None
    T = None
    flat = {}
    
    for k, v in nobs.items():
        if not isinstance(v, torch.Tensor):
            flat[k] = v
            continue
            
        if v.ndim >= 2:
            # 截取到 n_obs_steps
            v = v[:, :n_obs_steps, ...]
        
        if v.ndim == 5:
            # 图像: (B, T, C, H, W) -> (B*T, C, H, W)
            B, T = v.shape[0], v.shape[1]
            flat[k] = v.reshape(B * T, *v.shape[2:])
        elif v.ndim == 4:
            # 可能是已经没有时间维的图像 (B, C, H, W)，保持不变
            if B is None:
                B = v.shape[0]
            flat[k] = v
        elif v.ndim == 3:
            # 低维序列: (B, T, D) -> (B*T, D)
            B, T = v.shape[0], v.shape[1]
            flat[k] = v.reshape(B * T, v.shape[2])
        elif v.ndim == 2:
            # 低维无时间: (B, D)，保持不变
            if B is None:
                B = v.shape[0]
            flat[k] = v
        else:
            flat[k] = v
    
    feats = encoder(flat)
    return feats


class CombinedInferencePolicyLoader:
    """
    联合推理策略加载器
    用于从独立训练的检查点加载Action Predictor和Diffusion Policy
    """
    
    @staticmethod
    def load_from_checkpoints(
        action_predictor_ckpt_path: str,
        diffusion_policy_ckpt_path: str,
        device: str = 'cuda:0',
        use_diffusion_refinement: bool = True,
        feedback_to_predictor: bool = True,
    ) -> CombinedInferencePolicy:
        """
        从检查点加载联合推理策略
        
        Args:
            action_predictor_ckpt_path: Action Predictor检查点路径
            diffusion_policy_ckpt_path: Diffusion Policy检查点路径
            device: 设备
            use_diffusion_refinement: 是否使用diffusion refinement
            feedback_to_predictor: 是否反馈给predictor
        
        Returns:
            combined_policy: 联合推理策略
        """
        import dill
        
        # 加载Action Predictor
        ap_payload = torch.load(
            open(action_predictor_ckpt_path, 'rb'), 
            pickle_module=dill,
            map_location=device
        )
        action_predictor = ap_payload['cfg'].policy
        action_predictor.load_state_dict(ap_payload['state_dicts']['model'])
        action_predictor.eval()
        
        # 加载Diffusion Policy
        dp_payload = torch.load(
            open(diffusion_policy_ckpt_path, 'rb'), 
            pickle_module=dill,
            map_location=device
        )
        diffusion_policy = dp_payload['cfg'].policy
        diffusion_policy.load_state_dict(dp_payload['state_dicts']['model'])
        diffusion_policy.eval()
        
        # 创建联合策略
        combined_policy = CombinedInferencePolicy(
            action_predictor=action_predictor,
            diffusion_policy=diffusion_policy,
            use_diffusion_refinement=use_diffusion_refinement,
            feedback_to_predictor=feedback_to_predictor,
        )
        
        combined_policy.to(device)
        
        return combined_policy
    
    @staticmethod
    def load_from_hydra_checkpoints(
        action_predictor_ckpt_path: str,
        diffusion_policy_ckpt_path: str,
        device: str = 'cuda:0',
        use_diffusion_refinement: bool = True,
        feedback_to_predictor: bool = True,
    ) -> CombinedInferencePolicy:
        """
        从Hydra格式的检查点加载
        """
        import dill
        import hydra
        
        # 加载Action Predictor
        ap_payload = torch.load(
            open(action_predictor_ckpt_path, 'rb'), 
            pickle_module=dill,
            map_location=device
        )
        action_predictor = hydra.utils.instantiate(ap_payload['cfg'].policy)
        action_predictor.load_state_dict(ap_payload['state_dicts']['model'])
        action_predictor.eval()
        
        # 加载Diffusion Policy  
        dp_payload = torch.load(
            open(diffusion_policy_ckpt_path, 'rb'), 
            pickle_module=dill,
            map_location=device
        )
        
        # 使用Enhanced版本
        # 修改config中的policy target
        dp_cfg = dp_payload['cfg'].copy()
        dp_cfg.policy._target_ = 'diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy.EnhancedDiffusionUnetLowdimPolicy'
        
        diffusion_policy = hydra.utils.instantiate(dp_cfg.policy)
        diffusion_policy.load_state_dict(ap_payload['state_dicts']['model'], strict=False)
        diffusion_policy.eval()
        
        # 创建联合策略
        combined_policy = CombinedInferencePolicy(
            action_predictor=action_predictor,
            diffusion_policy=diffusion_policy,
            use_diffusion_refinement=use_diffusion_refinement,
            feedback_to_predictor=feedback_to_predictor,
        )
        
        combined_policy.to(device)
        
        return combined_policy
