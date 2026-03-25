"""
联合推理评估脚本
用于评估Combined Policy在环境中的表现
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    # 使用当前脚本所在目录作为根目录
    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)

import os
import copy
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pathlib
import dill
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.common.pytorch_util import dict_apply
from einops import rearrange

from diffusion_policy.policy.combined_inference_policy import CombinedInferencePolicy, attach_robust_predictor
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorLowdimPolicy

from diffusion_policy.policy.combined_inference_policy import CombinedInferencePolicy, attach_robust_predictor


# ==========================================
# 观测键名映射工具
# ==========================================
def create_obs_key_mapper(source_keys: list, target_keys: list):
    """
    创建观测键名映射函数。
    
    Args:
        source_keys: 环境/数据集提供的键名列表
        target_keys: 模型期望的键名列表
    
    Returns:
        映射函数 obs_dict -> mapped_obs_dict
    """
    # 尝试自动匹配（按顺序或按相似度）
    key_map = {}
    for src, tgt in zip(source_keys, target_keys):
        if src != tgt:
            key_map[src] = tgt
    
    if not key_map:
        return lambda x: x  # 无需映射
    
    print(f"  观测键映射: {key_map}")
    
    def mapper(obs_dict):
        result = {}
        for k, v in obs_dict.items():
            new_key = key_map.get(k, k)
            result[new_key] = v
        return result
    
    return mapper


def get_model_obs_keys(policy):
    """获取模型期望的观测键名列表"""
    # 尝试从 encoder 获取
    if hasattr(policy, 'encoder') and hasattr(policy.encoder, 'key_shape_map'):
        return list(policy.encoder.key_shape_map.keys())
    if hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'key_shape_map'):
        return list(policy.obs_encoder.key_shape_map.keys())
    if hasattr(policy, 'policy'):
        return get_model_obs_keys(policy.policy)
    return []


class ObsKeyMappingWrapper:
    """
    包装 policy，在 predict_action 前进行键名映射。
    """
    def __init__(self, policy, key_mapper):
        self.policy = policy
        self.key_mapper = key_mapper
        self.device = getattr(policy, 'device', 'cuda:0')
        self.dtype = getattr(policy, 'dtype', torch.float32)
    
    def predict_action(self, obs_dict, *args, **kwargs):
        mapped_obs = self.key_mapper(obs_dict)
        return self.policy.predict_action(mapped_obs, *args, **kwargs)
    
    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
    
    def __getattr__(self, name):
        return getattr(self.policy, name)



# ==========================================
# Enhanced Wrapper for Image Policy
# ==========================================
class EnhancedImagePolicyWrapper(nn.Module):
    """
    包装原始的 Image Diffusion Policy，为其添加基于初始轨迹的 Refinement 能力。
    不需要重新实例化模型，直接复用已加载的 policy 组件。
    """
    def __init__(self, policy, use_ddim=False, num_inference_steps=None):
        super().__init__()
        # self.policy 是一个 nn.Module，会被注册到 self._modules['policy']
        self.policy = policy
        self.device = policy.device
        self.dtype = policy.dtype
        
        # 引用核心组件
        self.model = policy.model
        self.obs_encoder = policy.obs_encoder
        self.normalizer = policy.normalizer
        self.noise_scheduler = policy.noise_scheduler
        
        # === 补全属性 ===
        self.horizon = getattr(policy, 'horizon', 16)
        self.action_dim = getattr(policy, 'action_dim', None)
        self.obs_dim = getattr(policy, 'obs_dim', 0)
        self.n_obs_steps = getattr(policy, 'n_obs_steps', 2)
        self.n_action_steps = getattr(policy, 'n_action_steps', 8)
        
        # 替换 Scheduler 为 DDIM (如果需要)
        if use_ddim:
            if not isinstance(self.noise_scheduler, DDIMScheduler):
                print("Swapping Noise Scheduler to DDIM...")
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type='epsilon'
                )
                if hasattr(self.policy, 'noise_scheduler'):
                    self.policy.noise_scheduler = self.noise_scheduler

        # 设置推理参数
        if num_inference_steps is None:
            if hasattr(self.noise_scheduler, 'config'):
                num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            else:
                num_inference_steps = 100
        self.num_inference_steps = num_inference_steps
        self.init_trajectory_steps = max(num_inference_steps // 4, 10)

    # === 属性代理 ===
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get('_modules')
            if modules is not None and 'policy' in modules:
                policy = modules['policy']
                return getattr(policy, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def reset(self):
        self.policy.reset()

    def eval(self):
        self.policy.eval()

    def to(self, device):
        self.policy.to(device)
        self.device = device
        return self

    def predict_action(self, obs_dict, init_action=None):
        """
        Args:
            obs_dict: 观测字典 - 支持两种格式:
                1) {modality_key: (B, T, C, H, W)} - runner 直接提供
                2) {'obs': {modality_key: ...}} - 原始格式
            init_action: 来自 Action Predictor 的初始动作 [B, T, D]
        """
        # 解包 obs_dict：如果是 {'obs': {...}} 格式，提取内部字典
        if 'obs' in obs_dict and isinstance(obs_dict['obs'], dict):
            obs_dict = obs_dict['obs']
        
        # 1. 归一化并准备维度信息
        # 只归一化 normalizer 中存在的键
        nobs = {}
        for key, value in obs_dict.items():
            if key in self.normalizer.params_dict:
                nobs[key] = self.normalizer[key].normalize(value)
            else:
                # 对于不在 normalizer 中的键（如图像），直接使用原值
                nobs[key] = value
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        # 2. 编码图像特征 - 参考 DiffusionUnetImagePolicy 的处理方式
        # 使用 dict_apply 展平时间维度: (B, T, C, H, W) -> (B*T, C, H, W)
        from diffusion_policy.common.pytorch_util import dict_apply
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]) if x.ndim >= 3 else x)
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, D)
        
        # reshape 回 (B, To*D) 作为 global_cond
        global_cond = nobs_features.reshape(B, -1)

        # 3. 准备初始轨迹（若提供）并归一化到 policy 的尺度
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
            init_trajectory = init_action_normalized

        # 4. 采样（使用 wrapper 的 conditional_sample）
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            global_cond=global_cond,
            init_trajectory=init_trajectory
        )

        # 5. 反归一化并切出 action
        # 增加裁剪保护，防止低步数下的数值爆炸导致 MuJoCo 崩溃
        nsample = torch.clamp(nsample, -1.5, 1.5)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = max(To - 1, 0)
        end = start + self.n_action_steps
        action = action_pred[:, start:end, :]

        return {
            'action': action,
            'action_pred': action_pred
        }

    def conditional_sample(self, 
            condition_data, condition_mask, 
            global_cond, init_trajectory=None, 
            generator=None, **kwargs):
        
        scheduler = self.noise_scheduler
        model = self.model
        
        if init_trajectory is not None:
            # === Refinement Mode ===
            num_inference_steps = self.init_trajectory_steps
            scheduler.set_timesteps(num_inference_steps)
            start_timestep = scheduler.timesteps[0].item()
            noise = torch.randn(
                size=init_trajectory.shape, 
                dtype=init_trajectory.dtype,
                device=init_trajectory.device,
                generator=generator
            )
            start_timestep_tensor = torch.tensor([start_timestep], device=init_trajectory.device)
            trajectory = scheduler.add_noise(init_trajectory, noise, start_timestep_tensor)
        else:
            # === Standard Mode ===
            num_inference_steps = self.num_inference_steps
            scheduler.set_timesteps(num_inference_steps)
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator
            )
            
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                sample=trajectory, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
            
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory


# ==========================================
# Loading Functions
# ==========================================

def load_action_predictor_image(ckpt_path: str, device: str = 'cuda:0'):
    """
    加载 Image Action Predictor，自动检测是 Transformer 还是 VAE 后端
    """
    payload = torch.load(
        open(ckpt_path, 'rb'),
        pickle_module=dill,
        map_location=device
    )
    cfg = payload['cfg']
    state_dict = payload['state_dicts']['model']
    
    # 检测是否为 VAE 后端（通过 state_dict key 判断）
    looks_like_vae = any('model.net.encoder_net' in k or 'model.net.decoder_net' in k for k in state_dict.keys())
    
    if looks_like_vae:
        # 使用 Image VAE Policy
        print("[INFO] Detected Image VAE backend, using ActionPredictorImageVAEPolicy")
        try:
            OmegaConf.set_struct(cfg.policy, False)
        except:
            pass
        
        # 修改 target 为 Image VAE Policy
        cfg.policy._target_ = 'diffusion_policy.policy.action_predictor_image_vae_policy.ActionPredictorImageVAEPolicy'
        
        # 如果 model 配置不存在或不是 VAEModel，重新构建
        if not hasattr(cfg.policy, 'model') or 'VAEModel' not in str(cfg.policy.model.get('_target_', '')):
            # 从现有配置推断参数
            encoder_output_dim = cfg.policy.get('encoder_output_dim', 128)
            action_dim = cfg.policy.get('action_dim', cfg.get('action_dim', 10))
            horizon = cfg.policy.get('horizon', cfg.get('horizon', 16))
            n_obs_steps = cfg.policy.get('n_obs_steps', cfg.get('n_obs_steps', 2))
            
            cfg.policy.model = OmegaConf.create({
                '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
                'action_dim': action_dim,
                'action_horizon': horizon,
                'obs_dim': encoder_output_dim,
                'obs_horizon': n_obs_steps,
                'latent_dim': 32,
                'layer': 256,
                'use_ema': True,
                'pretrain': False,
                'ckpt_path': None,
                # temperature 参数可能不被支持，移除
            })
    
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)
    
    # Attach robust wrapper to policy (moved to combined_inference_policy)
    attach_robust_predictor(policy)
    return policy, cfg


def load_diffusion_policy(ckpt_path: str, device: str = 'cuda:0', use_ema: bool = True, use_ddim: bool = False):
    """加载 Lowdim Diffusion Policy"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']

    model_config = cfg.policy.get('model', None)
    enhanced_cfg = OmegaConf.create({
        '_target_': 'diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy.EnhancedDiffusionUnetLowdimPolicy',
        'model': model_config,
        'noise_scheduler': cfg.policy.noise_scheduler,
        'horizon': cfg.policy.horizon,
        'obs_dim': cfg.policy.obs_dim,
        'action_dim': cfg.policy.action_dim,
        'n_action_steps': cfg.policy.n_action_steps,
        'n_obs_steps': cfg.policy.n_obs_steps,
        'num_inference_steps': cfg.policy.get('num_inference_steps', 100),
        'init_trajectory_steps': cfg.get('init_trajectory_steps', 25),
        'obs_as_local_cond': cfg.policy.get('obs_as_local_cond', False),
        'obs_as_global_cond': cfg.policy.get('obs_as_global_cond', True),
        'pred_action_steps_only': cfg.policy.get('pred_action_steps_only', False),
        'oa_step_convention': cfg.policy.get('oa_step_convention', True),
    })
    
    policy = hydra.utils.instantiate(enhanced_cfg)
    state_dict_key = 'ema_model' if use_ema and 'ema_model' in payload['state_dicts'] else 'model'
    policy.load_state_dict(payload['state_dicts'][state_dict_key], strict=False)
    policy.eval()
    policy.to(device)
    return policy, cfg


def load_diffusion_policy_image(ckpt_path: str, device: str = 'cuda:0', use_ema: bool = True, use_ddim: bool = False):
    """加载 Image Diffusion Policy 并使用 Wrapper"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']

    policy = hydra.utils.instantiate(cfg.policy)
    state_dict_key = 'ema_model' if use_ema and 'ema_model' in payload['state_dicts'] else 'model'
    policy.load_state_dict(payload['state_dicts'][state_dict_key], strict=False)
    policy.eval()
    policy.to(device)

    wrapped_policy = EnhancedImagePolicyWrapper(policy=policy, use_ddim=use_ddim)
    return wrapped_policy, cfg

def load_action_predictor(ckpt_path: str, device: str = 'cuda:0'):
    # Lowdim loading logic (unchanged)
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']
    try: OmegaConf.set_struct(cfg.policy, False)
    except: pass
    state_dict = payload['state_dicts']['model']
    looks_like_vae = any('net.encoder_net' in k for k in state_dict.keys())
    if looks_like_vae:
        cfg.policy.backend = 'vae'
        cfg.policy.model = {
            '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
            'action_dim': cfg.policy.action_dim,
            'action_horizon': cfg.policy.horizon,
            'obs_dim': cfg.policy.obs_dim,
            'obs_horizon': cfg.policy.n_obs_steps,
            'latent_dim': 32,
            'layer': 256,
            'use_ema': True,
            'pretrain': False,
            'ckpt_path': None,
        }
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)
    return policy, cfg


def parse_eval_seeds(eval_seeds):
    """解析命令行传入的 seed 列表。"""
    if eval_seeds is None:
        return None
    if isinstance(eval_seeds, (list, tuple)):
        seeds = [int(s) for s in eval_seeds]
    else:
        seeds = [int(s.strip()) for s in str(eval_seeds).split(',') if s.strip()]
    return seeds if len(seeds) > 0 else None


def aggregate_seed_dicts(seed_dicts):
    """聚合多个 seed 的字典结果，输出均值/标准差/格式化字符串。"""
    if len(seed_dicts) == 0:
        return {}
    if len(seed_dicts) == 1:
        return seed_dicts[0]

    aggregated = {
        'num_seeds': len(seed_dicts)
    }

    common_keys = None
    for d in seed_dicts:
        numeric_keys = set()
        for k, v in d.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)
        if common_keys is None:
            common_keys = numeric_keys
        else:
            common_keys = common_keys.intersection(numeric_keys)

    common_keys = common_keys or set()
    for key in sorted(common_keys):
        values = np.array([float(d[key]) for d in seed_dicts], dtype=np.float64)
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=0))
        aggregated[key] = mean_v
        aggregated[f'{key}_std'] = std_v
        aggregated[f'{key}_formatted'] = f'{mean_v:.4f}±{std_v:.4f}'

    return aggregated

def evaluate_combined_policy(
    action_predictor_ckpt: str,
    diffusion_policy_ckpt: str,
    output_dir: str,
    device: str = 'cuda:0',
    modality: str = 'lowdim',
    n_test: int = 50,
    n_test_vis: int = 4,
    use_diffusion_refinement: bool = True,
    feedback_to_predictor: bool = True,
    steps_list: list = None,
    use_ddim: bool = False,
    obs_key_map: dict = None,
    eval_seeds: list = None,
):
    if steps_list is None:
        steps_list = [100, 50, 25, 10]

    force_ddim = use_ddim or (len(steps_list) > 1)
    print(f"Modality: {modality}")
    
    print("Loading Action Predictor...")
    if modality == 'image':
        action_predictor, ap_cfg = load_action_predictor_image(action_predictor_ckpt, device)
    else:
        action_predictor, ap_cfg = load_action_predictor(action_predictor_ckpt, device)
    
    print("Loading Diffusion Policy...")
    if modality == 'image':
        diffusion_policy, dp_cfg = load_diffusion_policy_image(diffusion_policy_ckpt, device, use_ema=True, use_ddim=use_ddim)
    else:
        diffusion_policy, dp_cfg = load_diffusion_policy(diffusion_policy_ckpt, device, use_ema=True, use_ddim=use_ddim)
        
    original_noise_scheduler = copy.deepcopy(diffusion_policy.noise_scheduler)
    
    # 打印模型期望的观测键名，帮助诊断 KeyError
    print("\n=== 模型配置诊断 ===")
    ap_keys = get_model_obs_keys(action_predictor)
    dp_keys = get_model_obs_keys(diffusion_policy)
    if ap_keys:
        print(f"Action Predictor 期望的观测键: {ap_keys}")
    if dp_keys:
        print(f"Diffusion Policy 期望的观测键: {dp_keys}")
    
    # 检查两个模型的键是否一致
    if ap_keys and dp_keys and set(ap_keys) != set(dp_keys):
        print(f"⚠️  警告: 两个模型期望的观测键不一致!")
        print(f"   建议使用 --obs_key_map 参数进行映射")
    
    # 应用用户指定的键名映射
    key_mapper = None
    if obs_key_map:
        print(f"\n应用观测键映射: {obs_key_map}")
        def key_mapper(obs_dict):
            result = {}
            for k, v in obs_dict.items():
                new_key = obs_key_map.get(k, k)
                result[new_key] = v
            return result
    
    combined_policy = CombinedInferencePolicy(
        action_predictor=action_predictor,
        diffusion_policy=diffusion_policy,
        use_diffusion_refinement=use_diffusion_refinement,
        feedback_to_predictor=feedback_to_predictor,
    )
    combined_policy.set_normalizer(diffusion_policy.normalizer)
    
    # 如果有键名映射，包装 combined_policy
    if key_mapper:
        combined_policy = ObsKeyMappingWrapper(combined_policy, key_mapper)
    
    # 强制覆盖 EnvRunner 配置中的路径，防止 FileNotFoundError
    env_runner_cfg = dp_cfg.task.env_runner
    env_runner_cfg.n_test = n_test
    env_runner_cfg.n_test_vis = n_test_vis
    # 注意：不要硬编码数据集路径，使用配置文件中的路径
    # 如果需要覆盖，请确保路径与模型训练时使用的数据集匹配
    
    def create_env_runner(run_output_dir=None, test_seed=None):
        """创建新的 env_runner 实例"""
        cfg_local = copy.deepcopy(env_runner_cfg)
        if test_seed is not None:
            cfg_local.test_start_seed = int(test_seed)
        resolved_output_dir = os.path.abspath(run_output_dir or output_dir)
        # 某些并行 worker 会直接写入 output_dir/media，需提前创建避免竞态失败
        os.makedirs(resolved_output_dir, exist_ok=True)
        os.makedirs(os.path.join(resolved_output_dir, 'media'), exist_ok=True)
        return hydra.utils.instantiate(cfg_local, output_dir=resolved_output_dir)
    
    def safe_close_env_runner(runner):
        """安全关闭 env_runner"""
        try:
            if hasattr(runner, 'env') and runner.env is not None:
                runner.env.close()
        except Exception as e:
            print(f"Warning: Error closing env_runner: {e}")
    
    class DiffusionOnlyWrapper:
        def __init__(self, policy, key_mapper=None):
            self.policy = policy
            self.key_mapper = key_mapper
            self.device = getattr(policy, 'device', 'cuda:0')
            self.dtype = getattr(policy, 'dtype', torch.float32)
        def predict_action(self, obs_dict):
            if self.key_mapper:
                obs_dict = self.key_mapper(obs_dict)
            return self.policy.predict_action(obs_dict, init_action=None)
        def reset(self):
            if hasattr(self.policy, 'reset'): self.policy.reset()
    
    # 同样包装 action_predictor 用于单独评估
    if key_mapper:
        action_predictor_eval = ObsKeyMappingWrapper(action_predictor, key_mapper)
    else:
        action_predictor_eval = action_predictor
    
    diffusion_only_wrapper = DiffusionOnlyWrapper(diffusion_policy, key_mapper)
    all_results = {}

    if eval_seeds:
        print(f"Using multi-seed evaluation: {eval_seeds}")

    def run_policy_with_optional_seeds(policy_obj, policy_name, run_subdir, pre_run_hook=None, per_run_hook=None):
        """在单seed或多seed模式下运行评估。"""
        if not eval_seeds:
            if hasattr(policy_obj, 'reset'):
                policy_obj.reset()
            if pre_run_hook:
                pre_run_hook()
            env_runner = create_env_runner(run_output_dir=os.path.join(output_dir, run_subdir))
            try:
                result = env_runner.run(policy_obj)
            except Exception as e:
                print(f"Error in {policy_name} run: {e}")
                import traceback
                traceback.print_exc()
                result = {'test/mean_score': 0.0, 'error': str(e)}
            finally:
                safe_close_env_runner(env_runner)

            extra = per_run_hook() if per_run_hook else None
            return {'metrics': result, 'extra': extra}

        per_seed_metrics = []
        per_seed_extra = []
        for seed in eval_seeds:
            print(f"[{policy_name}] seed={seed}")
            if hasattr(policy_obj, 'reset'):
                policy_obj.reset()
            if pre_run_hook:
                pre_run_hook()

            seed_out_dir = os.path.join(output_dir, run_subdir, f'seed_{seed}')
            env_runner = create_env_runner(run_output_dir=seed_out_dir, test_seed=seed)
            try:
                result = env_runner.run(policy_obj)
            except Exception as e:
                print(f"Error in {policy_name} run (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                result = {'test/mean_score': 0.0, 'error': str(e)}
            finally:
                safe_close_env_runner(env_runner)

            result['eval/seed'] = int(seed)
            per_seed_metrics.append(result)
            if per_run_hook:
                extra = per_run_hook()
                if isinstance(extra, dict):
                    extra = dict(extra)
                    extra['eval/seed'] = int(seed)
                per_seed_extra.append(extra)

        aggregated_metrics = aggregate_seed_dicts(per_seed_metrics)
        aggregated_extra = aggregate_seed_dicts(per_seed_extra) if per_seed_extra else None
        return {
            'metrics': aggregated_metrics,
            'extra': aggregated_extra,
            'per_seed_metrics': per_seed_metrics,
            'per_seed_extra': per_seed_extra,
        }
    
    print("\n=== Action Predictor Only ===")
    ap_eval = run_policy_with_optional_seeds(
        policy_obj=action_predictor_eval,
        policy_name='Action Predictor',
        run_subdir='action_predictor_only'
    )
    ap_results = ap_eval['metrics']
    score_text = ap_results.get('test/mean_score_formatted', ap_results.get('test/mean_score', 'N/A'))
    print(f"Test mean score: {score_text}")
    all_results['action_predictor_only'] = ap_results
    if 'per_seed_metrics' in ap_eval:
        all_results['action_predictor_only_per_seed'] = ap_eval['per_seed_metrics']

    for steps in steps_list:
        print(f"\nEvaluating with inference steps: {steps}")
        step_results = {}

        if hasattr(diffusion_policy, 'num_inference_steps'):
            diffusion_policy.num_inference_steps = steps
        if hasattr(diffusion_policy, 'init_trajectory_steps'):
            diffusion_policy.init_trajectory_steps = steps

        diffusion_policy.noise_scheduler = copy.deepcopy(original_noise_scheduler)
        if force_ddim:
            ddim_scheduler = DDIMScheduler(
                num_train_timesteps=original_noise_scheduler.config.num_train_timesteps,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
            diffusion_policy.noise_scheduler = ddim_scheduler
            if hasattr(diffusion_policy, 'policy'):
                diffusion_policy.policy.noise_scheduler = ddim_scheduler
        
        print(f"\n--- Combined Policy (Refinement steps={steps}) ---")
        if hasattr(diffusion_policy, 'init_trajectory_steps'):
            diffusion_policy.init_trajectory_steps = steps
        combined_eval = run_policy_with_optional_seeds(
            policy_obj=combined_policy,
            policy_name='Combined',
            run_subdir=f'steps_{steps}/combined',
            pre_run_hook=(lambda: combined_policy.reset_timing_stats()) if hasattr(combined_policy, 'reset_timing_stats') else None,
            per_run_hook=lambda: combined_policy.get_inference_stats() if hasattr(combined_policy, 'get_inference_stats') else {}
        )
        combined_results = combined_eval['metrics']
        score_text = combined_results.get('test/mean_score_formatted', combined_results.get('test/mean_score', 'N/A'))
        print(f"Score: {score_text}")
        step_results['combined'] = {
            'metrics': combined_results,
            'timing': combined_eval.get('extra', {}) or {}
        }
        if 'per_seed_metrics' in combined_eval:
            step_results['combined_per_seed_metrics'] = combined_eval['per_seed_metrics']
            step_results['combined_per_seed_timing'] = combined_eval.get('per_seed_extra', [])
        
        print(f"\n--- Diffusion Only (Total steps={steps}) ---")
        if hasattr(diffusion_policy, 'num_inference_steps'):
            diffusion_policy.num_inference_steps = steps
        diffusion_eval = run_policy_with_optional_seeds(
            policy_obj=diffusion_only_wrapper,
            policy_name='Diffusion Only',
            run_subdir=f'steps_{steps}/diffusion_only'
        )
        diffusion_results = diffusion_eval['metrics']
        score_text = diffusion_results.get('test/mean_score_formatted', diffusion_results.get('test/mean_score', 'N/A'))
        print(f"Score: {score_text}")

        step_results['diffusion_only'] = {'metrics': diffusion_results}
        if 'per_seed_metrics' in diffusion_eval:
            step_results['diffusion_only_per_seed_metrics'] = diffusion_eval['per_seed_metrics']
        all_results[f'steps_{steps}'] = step_results

    import json
    results_path = os.path.join(output_dir, 'eval_results_comparison.json')
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Video': return {'type': 'Video'}
        return str(obj)
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")
    return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_predictor_ckpt', type=str, required=True)
    parser.add_argument('--diffusion_policy_ckpt', type=str, required=True)
    parser.add_argument('--modality', type=str, default='lowdim', choices=['lowdim', 'image'])
    parser.add_argument('--output_dir', type=str, default='data/outputs/eval_combined')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--n_test_vis', type=int, default=4)
    parser.add_argument('--no_refinement', action='store_true')
    parser.add_argument('--no_feedback', action='store_true')
    parser.add_argument('--steps_list', type=int, nargs='+', default=[100, 50, 25, 10])
    parser.add_argument('--use_ddim', action='store_true')
    parser.add_argument('--eval_seeds', type=int, nargs='+', default=None,
                        help='评估使用的随机种子列表，例如: --eval_seeds 100000 100100 100200')
    # 键名映射参数
    parser.add_argument('--obs_key_map', type=str, nargs='*', default=None,
                        help='观测键名映射，格式: src1:tgt1 src2:tgt2。例如: agentview_image:sideview_image')
    args = parser.parse_args()
    
    # 解析键名映射
    obs_key_map = None
    if args.obs_key_map:
        obs_key_map = {}
        for mapping in args.obs_key_map:
            if ':' in mapping:
                src, tgt = mapping.split(':', 1)
                obs_key_map[src] = tgt
        print(f"用户指定的观测键映射: {obs_key_map}")

    eval_seeds = parse_eval_seeds(args.eval_seeds)
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_combined_policy(
        action_predictor_ckpt=args.action_predictor_ckpt,
        diffusion_policy_ckpt=args.diffusion_policy_ckpt,
        output_dir=args.output_dir,
        device=args.device,
        modality=args.modality,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        use_diffusion_refinement=not args.no_refinement,
        feedback_to_predictor=not args.no_feedback,
        steps_list=args.steps_list,
        use_ddim=args.use_ddim,
        obs_key_map=obs_key_map,
        eval_seeds=eval_seeds,
    )

if __name__ == "__main__":
    main()