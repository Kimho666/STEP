"""
CFM 联合推理评估脚本

评估 VAE + CFM 联合推理的效果，特别是 1 步 CFM 的精度提升

Usage:
    # 基础评估（VAE + CFM 1步）
    python eval_cfm_combined.py \
        --ap-checkpoint checkpoints/vae/latest.ckpt \
        --cfm-checkpoint checkpoints/cfm/latest.ckpt \
        -o output/combined_eval

    # 调整 start_t（控制初始轨迹的噪声程度）
    python eval_cfm_combined.py \
        --ap-checkpoint vae.ckpt --cfm-checkpoint cfm.ckpt \
        -o output --start-t 0.3

    # 对比不同配置
    python eval_cfm_combined.py \
        --ap-checkpoint vae.ckpt --cfm-checkpoint cfm.ckpt \
        -o output --compare-configs
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import time
import numpy as np
import copy
from typing import Optional, List, Dict

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.cfm_combined_inference_policy import CFMCombinedInferencePolicy


class InferenceTimeTracker:
    """追踪推理时间"""
    def __init__(self, policy):
        self.policy = policy
        self.inference_times = []
        self._original_predict_action = policy.predict_action
        
        def timed_predict_action(obs_dict, *args, **kwargs):
            start = time.perf_counter()
            result = self._original_predict_action(obs_dict, *args, **kwargs)
            end = time.perf_counter()
            self.inference_times.append(end - start)
            return result
        
        policy.predict_action = timed_predict_action
    
    def get_stats(self):
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'total_inference_time': 0.0}
        
        times = np.array(self.inference_times)
        times_ex_first = times[1:] if len(times) > 1 else times
        
        return {
            'avg_inference_time': float(np.mean(times_ex_first)) if len(times_ex_first) > 0 else 0.0,
            'total_inference_time': float(np.sum(times)),
            'min_inference_time': float(np.min(times_ex_first)) if len(times_ex_first) > 0 else 0.0,
            'max_inference_time': float(np.max(times_ex_first)) if len(times_ex_first) > 0 else 0.0,
            'total_inference_calls': len(times),
            'first_inference_time': float(times[0]) if len(times) > 0 else 0.0,
        }
    
    def restore(self):
        self.policy.predict_action = self._original_predict_action


def load_action_predictor(ckpt_path: str, device: str = 'cuda:0'):
    """
    加载 Action Predictor，参考 eval_combined_inference.py 的加载方式
    支持 Transformer 和 VAE 后端
    """
    from omegaconf import OmegaConf
    
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']
    
    try:
        OmegaConf.set_struct(cfg.policy, False)
    except:
        pass
    
    state_dict = payload['state_dicts']['model']
    
    # 检测是否为 VAE 后端
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
    
    print(f"[INFO] Loaded Action Predictor:")
    print(f"  action_dim: {policy.action_dim}")
    print(f"  n_action_steps: {policy.n_action_steps}")
    print(f"  horizon: {policy.horizon}")
    
    return policy, cfg


def load_policy_from_checkpoint(checkpoint_path: str, device: str = 'cuda:0', use_ema: bool = True):
    """从检查点加载策略"""
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    
    try:
        workspace = cls(cfg, output_dir='.')
    except TypeError:
        workspace = cls(cfg)
    
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if use_ema and cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg, workspace


def run_evaluation(
    combined_policy: CFMCombinedInferencePolicy,
    cfg,
    output_dir: str,
    config_name: str = "default"
) -> Dict:
    """运行评估"""
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    
    tracker = InferenceTimeTracker(combined_policy)
    combined_policy.reset()
    
    print(f"\n开始评估 ({config_name})...")
    start_time = time.perf_counter()
    
    runner_log = env_runner.run(combined_policy)
    
    end_time = time.perf_counter()
    total_wall_time = end_time - start_time
    
    inference_stats = tracker.get_stats()
    tracker.restore()
    
    # 获取联合推理统计
    combined_stats = combined_policy.get_inference_stats()
    
    runner_log['eval/total_wall_time'] = total_wall_time
    runner_log['eval/config_name'] = config_name
    runner_log.update({f'eval/{k}': v for k, v in inference_stats.items()})
    runner_log.update({f'combined/{k}': v for k, v in combined_stats.items()})
    
    return runner_log


def parse_eval_seeds(eval_seeds: Optional[str]) -> Optional[List[int]]:
    """解析多 seed 参数。"""
    if eval_seeds is None:
        return None
    seeds = [int(s.strip()) for s in eval_seeds.split(',') if s.strip()]
    if len(seeds) == 0:
        return None
    return seeds


def aggregate_seed_logs(seed_logs: List[Dict]) -> Dict:
    """对多个 seed 的评估结果做均值和标准差聚合。"""
    if len(seed_logs) == 0:
        return {}
    if len(seed_logs) == 1:
        return seed_logs[0]

    numeric_key_sets = []
    for log in seed_logs:
        numeric_keys = set()
        for k, v in log.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)
        numeric_key_sets.append(numeric_keys)

    common_numeric_keys = set.intersection(*numeric_key_sets) if numeric_key_sets else set()

    aggregated = {
        'eval/num_seeds': len(seed_logs),
        'eval/seeds': [log.get('eval/seed', None) for log in seed_logs],
    }

    for key in sorted(common_numeric_keys):
        values = np.array([float(log[key]) for log in seed_logs], dtype=np.float64)
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=0))

        # 保留原 key 为均值，兼容现有打印/下游逻辑
        aggregated[key] = mean_v
        aggregated[f'{key}_std'] = std_v
        aggregated[f'{key}_formatted'] = f"{mean_v:.4f}±{std_v:.4f}"

    return aggregated


def run_evaluation_multi_seed(
    combined_policy: CFMCombinedInferencePolicy,
    cfg,
    output_dir: str,
    config_name: str,
    eval_seeds: Optional[List[int]] = None,
) -> Dict:
    """支持多 seed 评估并自动聚合结果。"""
    if not eval_seeds:
        return run_evaluation(combined_policy, cfg, output_dir, config_name)

    seed_logs = []
    for seed in eval_seeds:
        print(f"\\n[Multi-Seed] Running seed={seed} for {config_name}")
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.task.env_runner.test_start_seed = int(seed)

        seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
        pathlib.Path(seed_output_dir).mkdir(parents=True, exist_ok=True)

        seed_config_name = f"{config_name}_seed{seed}"
        runner_log = run_evaluation(combined_policy, cfg_seed, seed_output_dir, seed_config_name)
        runner_log['eval/seed'] = int(seed)
        save_results(runner_log, seed_output_dir, seed_config_name)
        seed_logs.append(runner_log)

    aggregated = aggregate_seed_logs(seed_logs)
    return aggregated


def run_raw_policy_evaluation(
    policy,
    cfg,
    output_dir: str,
    config_name: str = "default"
) -> Dict:
    """运行非 CombinedPolicy 的评估（如 CFM-only）。"""
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    tracker = InferenceTimeTracker(policy)
    if hasattr(policy, 'reset'):
        policy.reset()

    start_time = time.perf_counter()
    runner_log = env_runner.run(policy)
    total_wall_time = time.perf_counter() - start_time

    inference_stats = tracker.get_stats()
    tracker.restore()

    runner_log['eval/total_wall_time'] = total_wall_time
    runner_log['eval/config_name'] = config_name
    runner_log.update({f'eval/{k}': v for k, v in inference_stats.items()})
    return runner_log


def run_raw_policy_evaluation_multi_seed(
    policy,
    cfg,
    output_dir: str,
    config_name: str,
    eval_seeds: Optional[List[int]] = None,
) -> Dict:
    """支持多 seed 的非 CombinedPolicy 评估。"""
    if not eval_seeds:
        return run_raw_policy_evaluation(policy, cfg, output_dir, config_name)

    seed_logs = []
    for seed in eval_seeds:
        print(f"\\n[Multi-Seed] Running seed={seed} for {config_name}")
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.task.env_runner.test_start_seed = int(seed)

        seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
        pathlib.Path(seed_output_dir).mkdir(parents=True, exist_ok=True)

        seed_config_name = f"{config_name}_seed{seed}"
        runner_log = run_raw_policy_evaluation(policy, cfg_seed, seed_output_dir, seed_config_name)
        runner_log['eval/seed'] = int(seed)
        save_results(runner_log, seed_output_dir, seed_config_name)
        seed_logs.append(runner_log)

    aggregated = aggregate_seed_logs(seed_logs)
    return aggregated


def print_summary(runner_log: Dict, config_name: str):
    """打印评估摘要"""
    print(f"\n{'='*70}")
    print(f"评估完成: {config_name}")
    print(f"{'='*70}")
    
    if 'test/mean_score' in runner_log:
        print(f"测试得分: {runner_log['test/mean_score']:.4f}")
    if 'test/mean_score_formatted' in runner_log:
        print(f"测试得分(均值±方差): {runner_log['test/mean_score_formatted']}")
    
    print(f"\n推理时间统计:")
    print(f"  总评估时间: {runner_log['eval/total_wall_time']:.4f} 秒")
    print(f"  平均推理时间: {runner_log.get('eval/avg_inference_time', 0)*1000:.3f} ms")
    
    if 'combined/avg_predictor_time' in runner_log:
        print(f"\n联合推理细节:")
        print(f"  start_t: {runner_log.get('combined/start_t', 'N/A')}")
        print(f"  refinement_steps: {runner_log.get('combined/refinement_steps', 'N/A')}")
        print(f"  平均 Predictor 时间: {runner_log.get('combined/avg_predictor_time', 0)*1000:.3f} ms")
        print(f"  平均 CFM 时间: {runner_log.get('combined/avg_cfm_time', 0)*1000:.3f} ms")
    if 'combined/avg_predictor_time_formatted' in runner_log:
        print(f"  Predictor 时间(均值±方差): {runner_log.get('combined/avg_predictor_time_formatted', 'N/A')}")
    if 'combined/avg_cfm_time_formatted' in runner_log:
        print(f"  CFM 时间(均值±方差): {runner_log.get('combined/avg_cfm_time_formatted', 'N/A')}")
    
    print(f"{'='*70}\n")


def save_results(runner_log: Dict, output_dir: str, config_name: str):
    """保存结果"""
    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    
    out_path = os.path.join(output_dir, f'eval_log_{config_name}.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"结果已保存至: {out_path}")


@click.command()
@click.option('--ap-checkpoint', required=True, help='Action Predictor (VAE) 检查点路径')
@click.option('--cfm-checkpoint', required=True, help='CFM Policy 检查点路径')
@click.option('-o', '--output_dir', required=True, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='计算设备')
@click.option('--start-t', default=0.2, type=float, help='初始轨迹位置 (0.1-0.3 推荐)')
@click.option('--refinement-steps', default=1, type=int, help='CFM 精细化步数')
@click.option('--use-ema/--no-ema', default=True, help='是否使用 EMA 模型')
@click.option('--compare-configs', is_flag=True, help='对比多种配置')
@click.option('--compare-steps', default=None, type=str, help='对比不同 CFM 步数，逗号分隔，如 "1,2,3,5,10"')
@click.option('--compare-start-t', default=None, type=str, help='对比不同 start_t，逗号分隔，如 "0.1,0.2,0.3,0.5"')
@click.option('--no-refinement', is_flag=True, help='不使用 CFM refinement (仅 VAE)')
@click.option('--eval-seeds', default=None, type=str, help='评估使用的随机种子列表，逗号分隔，如 "100000,100100,100200"')
def main(ap_checkpoint, cfm_checkpoint, output_dir, device, start_t, refinement_steps, 
         use_ema, compare_configs, compare_steps, compare_start_t, no_refinement, eval_seeds):
    """
    CFM 联合推理评估
    
    核心思路：VAE 提供初始轨迹 → CFM 从 t=start_t 开始 1 步精细化
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型 - 使用专门的加载函数
    print("加载 Action Predictor...")
    action_predictor, cfg_ap = load_action_predictor(ap_checkpoint, device)
    
    print("加载 CFM Policy...")
    cfm_policy, cfg_cfm, workspace_cfm = load_policy_from_checkpoint(cfm_checkpoint, device, use_ema)

    seeds_list = parse_eval_seeds(eval_seeds)
    if seeds_list:
        print(f"[INFO] 启用多 seed 评估: {seeds_list}")
    
    # 对比不同 CFM 步数（联合推理）
    if compare_steps:
        steps_list = [int(s.strip()) for s in compare_steps.split(',')]
        print(f"\n对比联合推理不同 CFM 步数: {steps_list}")
        
        comparison_results = []
        
        # 先测试纯 AP（无 CFM refinement）
        print("\n" + "=" * 70)
        print("评估: AP_only (无 CFM refinement)")
        print("=" * 70)
        
        config_output_dir = os.path.join(output_dir, "AP_only")
        pathlib.Path(config_output_dir).mkdir(parents=True, exist_ok=True)
        
        combined_policy = CFMCombinedInferencePolicy(
            action_predictor=action_predictor,
            cfm_policy=cfm_policy,
            start_t=0.0,
            refinement_steps=0,
            use_cfm_refinement=False,
        )
        combined_policy.set_normalizer(cfm_policy.normalizer)
        
        runner_log = run_evaluation_multi_seed(
            combined_policy,
            cfg_cfm,
            config_output_dir,
            "AP_only",
            eval_seeds=seeds_list,
        )
        print_summary(runner_log, "AP_only")
        save_results(runner_log, config_output_dir, "AP_only")
        
        comparison_results.append({
            'config': 'AP_only',
            'start_t': 0.0,
            'steps': 0,
            'mean_score': runner_log.get('test/mean_score', 0),
            'avg_inference_time_ms': runner_log.get('eval/avg_inference_time', 0) * 1000,
            'total_wall_time': runner_log.get('eval/total_wall_time', 0),
        })
        
        # 测试不同 CFM 步数
        for steps in steps_list:
            config_name = f"AP+CFM_t{start_t}_{steps}step"
            print(f"\n{'=' * 70}")
            print(f"评估: {config_name}")
            print(f"{'=' * 70}")
            
            config_output_dir = os.path.join(output_dir, config_name)
            pathlib.Path(config_output_dir).mkdir(parents=True, exist_ok=True)
            
            combined_policy = CFMCombinedInferencePolicy(
                action_predictor=action_predictor,
                cfm_policy=cfm_policy,
                start_t=start_t,
                refinement_steps=steps,
                use_cfm_refinement=True,
            )
            combined_policy.set_normalizer(cfm_policy.normalizer)
            
            runner_log = run_evaluation_multi_seed(
                combined_policy,
                cfg_cfm,
                config_output_dir,
                config_name,
                eval_seeds=seeds_list,
            )
            print_summary(runner_log, config_name)
            save_results(runner_log, config_output_dir, config_name)
            
            comparison_results.append({
                'config': config_name,
                'start_t': start_t,
                'steps': steps,
                'mean_score': runner_log.get('test/mean_score', 0),
                'avg_inference_time_ms': runner_log.get('eval/avg_inference_time', 0) * 1000,
                'total_wall_time': runner_log.get('eval/total_wall_time', 0),
            })
        
        # 打印对比表格
        print("\n" + "=" * 100)
        print("CFM 步数对比摘要 (联合推理)")
        print("=" * 100)
        print(f"{'配置':<30} {'start_t':<10} {'步数':<8} {'平均得分':<12} {'推理时间(ms)':<15} {'总耗时(s)':<12}")
        print("-" * 100)
        for r in comparison_results:
            print(f"{r['config']:<30} {r['start_t']:<10.2f} {r['steps']:<8} {r['mean_score']:<12.4f} {r['avg_inference_time_ms']:<15.3f} {r['total_wall_time']:<12.2f}")
        print("=" * 100)
        
        # 保存对比结果
        comparison_path = os.path.join(output_dir, 'steps_comparison_results.json')
        json.dump(comparison_results, open(comparison_path, 'w'), indent=2)
        print(f"\n对比结果已保存至: {comparison_path}")
        
        # 生成报告
        report_path = os.path.join(output_dir, 'steps_comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("CFM 步数对比摘要 (Action Predictor + CFM 联合推理)\n")
            f.write("=" * 100 + "\n")
            f.write(f"start_t: {start_t}\n")
            f.write(f"对比步数: {steps_list}\n\n")
            f.write(f"{'配置':<30} {'步数':<8} {'平均得分':<12} {'推理时间(ms)':<15}\n")
            f.write("-" * 100 + "\n")
            for r in comparison_results:
                f.write(f"{r['config']:<30} {r['steps']:<8} {r['mean_score']:<12.4f} {r['avg_inference_time_ms']:<15.3f}\n")
            f.write("=" * 100 + "\n")
        print(f"报告已保存至: {report_path}")
        
        return
    
    # 对比不同 start_t
    if compare_start_t:
        start_t_list = [float(s.strip()) for s in compare_start_t.split(',')]
        print(f"\n对比联合推理不同 start_t: {start_t_list}")
        
        comparison_results = []
        
        for st in start_t_list:
            config_name = f"AP+CFM_t{st}_{refinement_steps}step"
            print(f"\n{'=' * 70}")
            print(f"评估: {config_name}")
            print(f"{'=' * 70}")
            
            config_output_dir = os.path.join(output_dir, config_name)
            pathlib.Path(config_output_dir).mkdir(parents=True, exist_ok=True)
            
            combined_policy = CFMCombinedInferencePolicy(
                action_predictor=action_predictor,
                cfm_policy=cfm_policy,
                start_t=st,
                refinement_steps=refinement_steps,
                use_cfm_refinement=True,
            )
            combined_policy.set_normalizer(cfm_policy.normalizer)
            
            runner_log = run_evaluation_multi_seed(
                combined_policy,
                cfg_cfm,
                config_output_dir,
                config_name,
                eval_seeds=seeds_list,
            )
            print_summary(runner_log, config_name)
            save_results(runner_log, config_output_dir, config_name)
            
            comparison_results.append({
                'config': config_name,
                'start_t': st,
                'steps': refinement_steps,
                'mean_score': runner_log.get('test/mean_score', 0),
                'avg_inference_time_ms': runner_log.get('eval/avg_inference_time', 0) * 1000,
                'total_wall_time': runner_log.get('eval/total_wall_time', 0),
            })
        
        # 打印对比表格
        print("\n" + "=" * 100)
        print("start_t 对比摘要 (联合推理)")
        print("=" * 100)
        print(f"{'配置':<30} {'start_t':<10} {'步数':<8} {'平均得分':<12} {'推理时间(ms)':<15}")
        print("-" * 100)
        for r in comparison_results:
            print(f"{r['config']:<30} {r['start_t']:<10.2f} {r['steps']:<8} {r['mean_score']:<12.4f} {r['avg_inference_time_ms']:<15.3f}")
        print("=" * 100)
        
        # 保存对比结果
        comparison_path = os.path.join(output_dir, 'start_t_comparison_results.json')
        json.dump(comparison_results, open(comparison_path, 'w'), indent=2)
        print(f"\n对比结果已保存至: {comparison_path}")
        
        return
    
    if compare_configs:
        # 对比多种配置
        configs = [
            # (name, start_t, refinement_steps, use_cfm)
            ("VAE_only", 0.0, 0, False),
            ("CFM_only_1step", None, None, "cfm_only_1"),
            ("CFM_only_10step", None, None, "cfm_only_10"),
            ("VAE+CFM_t0.1_1step", 0.1, 1, True),
            ("VAE+CFM_t0.2_1step", 0.2, 1, True),
            ("VAE+CFM_t0.3_1step", 0.3, 1, True),
            ("VAE+CFM_t0.2_2step", 0.2, 2, True),
        ]
        
        comparison_results = []
        
        for config in configs:
            name = config[0]
            config_output_dir = os.path.join(output_dir, name)
            pathlib.Path(config_output_dir).mkdir(parents=True, exist_ok=True)
            
            if config[2] == "cfm_only_1":
                # 纯 CFM 1 步
                cfm_policy.num_inference_steps = 1
                runner_log = run_raw_policy_evaluation_multi_seed(
                    cfm_policy,
                    cfg_cfm,
                    config_output_dir,
                    name,
                    eval_seeds=seeds_list,
                )
                
            elif config[2] == "cfm_only_10":
                # 纯 CFM 10 步
                cfm_policy.num_inference_steps = 10
                runner_log = run_raw_policy_evaluation_multi_seed(
                    cfm_policy,
                    cfg_cfm,
                    config_output_dir,
                    name,
                    eval_seeds=seeds_list,
                )
                
            else:
                # 联合推理
                combined_policy = CFMCombinedInferencePolicy(
                    action_predictor=action_predictor,
                    cfm_policy=cfm_policy,
                    start_t=config[1],
                    refinement_steps=config[2] if config[2] else 0,
                    use_cfm_refinement=config[3] if isinstance(config[3], bool) else True,
                )
                combined_policy.set_normalizer(cfm_policy.normalizer)
                
                runner_log = run_evaluation_multi_seed(
                    combined_policy,
                    cfg_cfm,
                    config_output_dir,
                    name,
                    eval_seeds=seeds_list,
                )
            
            print_summary(runner_log, name)
            save_results(runner_log, config_output_dir, name)
            
            comparison_results.append({
                'config': name,
                'mean_score': runner_log.get('test/mean_score', 0),
                'avg_inference_time_ms': runner_log.get('eval/avg_inference_time', 0) * 1000,
                'total_wall_time': runner_log.get('eval/total_wall_time', 0),
            })
        
        # 打印对比表格
        print("\n" + "=" * 90)
        print("配置对比摘要")
        print("=" * 90)
        print(f"{'配置':<25} {'平均得分':<12} {'平均推理时间(ms)':<20} {'总耗时(s)':<12}")
        print("-" * 90)
        for r in comparison_results:
            print(f"{r['config']:<25} {r['mean_score']:<12.4f} {r['avg_inference_time_ms']:<20.3f} {r['total_wall_time']:<12.2f}")
        print("=" * 90)
        
        # 保存对比结果
        comparison_path = os.path.join(output_dir, 'comparison_results.json')
        json.dump(comparison_results, open(comparison_path, 'w'), indent=2)
        print(f"\n对比结果已保存至: {comparison_path}")
        
    else:
        # 单次评估
        combined_policy = CFMCombinedInferencePolicy(
            action_predictor=action_predictor,
            cfm_policy=cfm_policy,
            start_t=start_t,
            refinement_steps=refinement_steps,
            use_cfm_refinement=not no_refinement,
        )
        combined_policy.set_normalizer(cfm_policy.normalizer)
        
        config_name = f"start_t{start_t}_steps{refinement_steps}"
        if no_refinement:
            config_name = "VAE_only"
        
        runner_log = run_evaluation_multi_seed(
            combined_policy,
            cfg_cfm,
            output_dir,
            config_name,
            eval_seeds=seeds_list,
        )
        print_summary(runner_log, config_name)
        save_results(runner_log, output_dir, config_name)


if __name__ == '__main__':
    main()
