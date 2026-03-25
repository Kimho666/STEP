"""
CFM (Conditional Flow Matching) 专属评估脚本

支持评估:
- CFMUnetLowdimPolicy (U-Net + Flow Matching)
- CFMTransformerLowdimPolicy (Transformer + Flow Matching, π0.5-style)

Usage:
    # 基础评估
    python eval_cfm.py -c checkpoints/cfm_unet/latest.ckpt -o output/cfm_eval

    # 调整推理步数（CFM优势：更少步数也能获得好结果）
    python eval_cfm.py -c checkpoints/cfm_transformer/latest.ckpt -o output/cfm_eval --steps 5

    # 对比不同推理步数
    python eval_cfm.py -c checkpoints/cfm_unet/latest.ckpt -o output/cfm_eval --compare-steps 1,5,10,20,50

    # 使用不同 ODE solver
    python eval_cfm.py -c checkpoints/cfm_transformer/latest.ckpt -o output/cfm_eval --solver heun --steps 10
"""

import sys
# use line-buffering for both stdout and stderr
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
from typing import Optional, List
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def patch_cfm_policy_solver(policy, solver: str, num_inference_steps: int):
    """
    为 CFM Policy 动态修改 ODE solver 和推理步数
    
    支持的 solver:
    - euler: 标准欧拉方法 (1阶)
    - heun: Heun方法 (2阶, 更精确但需要更多计算)
    - rk4: 4阶龙格-库塔 (4阶, 最精确)
    - midpoint: 中点法 (2阶)
    """
    policy.num_inference_steps = num_inference_steps
    
    if solver == 'euler':
        # 默认实现已经是 Euler，无需修改
        return policy
    
    # 为更高阶 solver 注入新的采样方法
    original_sample = policy.conditional_sample
    
    if solver == 'heun':
        def heun_sample(self_policy, **kwargs):
            """Heun (改进欧拉) ODE solver"""
            return _heun_solver(self_policy, **kwargs)
        policy.conditional_sample = lambda **kw: heun_sample(policy, **kw)
        
    elif solver == 'rk4':
        def rk4_sample(self_policy, **kwargs):
            """RK4 ODE solver"""
            return _rk4_solver(self_policy, **kwargs)
        policy.conditional_sample = lambda **kw: rk4_sample(policy, **kw)
        
    elif solver == 'midpoint':
        def midpoint_sample(self_policy, **kwargs):
            """Midpoint ODE solver"""
            return _midpoint_solver(self_policy, **kwargs)
        policy.conditional_sample = lambda **kw: midpoint_sample(policy, **kw)
    
    return policy


def _get_velocity(policy, x_t, t_current, cond=None, local_cond=None, global_cond=None):
    """统一获取速度场预测"""
    device = x_t.device
    batch_size = x_t.shape[0]
    
    t_tensor = torch.full(
        (batch_size,), 
        t_current * 1000,
        device=device,
        dtype=torch.long
    )
    
    # 判断是 Transformer (使用 cond) 还是 U-Net (使用 local/global_cond)
    if hasattr(policy, 'obs_as_cond') and policy.obs_as_cond:
        # Transformer policy
        pred_v = policy.model(
            sample=x_t, 
            timestep=t_tensor, 
            cond=cond
        )
    else:
        # U-Net policy
        pred_v = policy.model(
            x_t, 
            t_tensor, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
    
    return pred_v


def _heun_solver(policy, cond=None, local_cond=None, global_cond=None, 
                 condition_data=None, condition_mask=None,
                 num_inference_steps=None, generator=None,
                 batch_size=1, device=None, dtype=None):
    """
    Heun (改进欧拉/显式梯形) ODE solver
    
    k1 = f(t, x)
    k2 = f(t + dt, x + dt * k1)
    x_new = x + dt/2 * (k1 + k2)
    """
    if num_inference_steps is None:
        num_inference_steps = policy.num_inference_steps
    
    if device is None:
        device = policy.device
    if dtype is None:
        dtype = policy.dtype
    
    # 确定输出形状
    if hasattr(policy, 'pred_action_steps_only') and policy.pred_action_steps_only:
        shape = (batch_size, policy.n_action_steps, policy.action_dim)
    else:
        shape = (batch_size, policy.horizon, policy.action_dim)
    
    if condition_data is not None:
        shape = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype
    
    # 初始化
    x_t = torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    dt = -1.0 / num_inference_steps
    
    for i in range(num_inference_steps):
        t_current = 1.0 - i / num_inference_steps
        t_next = t_current + dt
        
        # 条件掩码处理
        if condition_mask is not None:
            x_t[condition_mask] = condition_data[condition_mask]
        
        # k1 = v(x_t, t)
        k1 = _get_velocity(policy, x_t, t_current, cond, local_cond, global_cond)
        
        # x_pred = x_t + dt * k1
        x_pred = x_t + k1 * dt
        
        # k2 = v(x_pred, t + dt)
        k2 = _get_velocity(policy, x_pred, max(t_next, 0.0), cond, local_cond, global_cond)
        
        # Heun update
        x_t = x_t + dt * 0.5 * (k1 + k2)
    
    if condition_mask is not None:
        x_t[condition_mask] = condition_data[condition_mask]
    
    return x_t


def _midpoint_solver(policy, cond=None, local_cond=None, global_cond=None,
                     condition_data=None, condition_mask=None,
                     num_inference_steps=None, generator=None,
                     batch_size=1, device=None, dtype=None):
    """
    中点法 ODE solver
    
    k1 = f(t, x)
    k2 = f(t + dt/2, x + dt/2 * k1)
    x_new = x + dt * k2
    """
    if num_inference_steps is None:
        num_inference_steps = policy.num_inference_steps
    
    if device is None:
        device = policy.device
    if dtype is None:
        dtype = policy.dtype
    
    if hasattr(policy, 'pred_action_steps_only') and policy.pred_action_steps_only:
        shape = (batch_size, policy.n_action_steps, policy.action_dim)
    else:
        shape = (batch_size, policy.horizon, policy.action_dim)
    
    if condition_data is not None:
        shape = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype
    
    x_t = torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    dt = -1.0 / num_inference_steps
    
    for i in range(num_inference_steps):
        t_current = 1.0 - i / num_inference_steps
        t_mid = t_current + dt / 2
        
        if condition_mask is not None:
            x_t[condition_mask] = condition_data[condition_mask]
        
        # k1 = v(x_t, t)
        k1 = _get_velocity(policy, x_t, t_current, cond, local_cond, global_cond)
        
        # x_mid = x_t + dt/2 * k1
        x_mid = x_t + k1 * (dt / 2)
        
        # k2 = v(x_mid, t + dt/2)
        k2 = _get_velocity(policy, x_mid, max(t_mid, 0.0), cond, local_cond, global_cond)
        
        # Midpoint update
        x_t = x_t + dt * k2
    
    if condition_mask is not None:
        x_t[condition_mask] = condition_data[condition_mask]
    
    return x_t


def _rk4_solver(policy, cond=None, local_cond=None, global_cond=None,
                condition_data=None, condition_mask=None,
                num_inference_steps=None, generator=None,
                batch_size=1, device=None, dtype=None):
    """
    4阶龙格-库塔 ODE solver
    
    k1 = f(t, x)
    k2 = f(t + dt/2, x + dt/2 * k1)
    k3 = f(t + dt/2, x + dt/2 * k2)
    k4 = f(t + dt, x + dt * k3)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    if num_inference_steps is None:
        num_inference_steps = policy.num_inference_steps
    
    if device is None:
        device = policy.device
    if dtype is None:
        dtype = policy.dtype
    
    if hasattr(policy, 'pred_action_steps_only') and policy.pred_action_steps_only:
        shape = (batch_size, policy.n_action_steps, policy.action_dim)
    else:
        shape = (batch_size, policy.horizon, policy.action_dim)
    
    if condition_data is not None:
        shape = condition_data.shape
        device = condition_data.device
        dtype = condition_data.dtype
    
    x_t = torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    dt = -1.0 / num_inference_steps
    
    for i in range(num_inference_steps):
        t_current = 1.0 - i / num_inference_steps
        t_mid = t_current + dt / 2
        t_next = t_current + dt
        
        if condition_mask is not None:
            x_t[condition_mask] = condition_data[condition_mask]
        
        # k1
        k1 = _get_velocity(policy, x_t, t_current, cond, local_cond, global_cond)
        
        # k2
        x_k2 = x_t + k1 * (dt / 2)
        k2 = _get_velocity(policy, x_k2, max(t_mid, 0.0), cond, local_cond, global_cond)
        
        # k3
        x_k3 = x_t + k2 * (dt / 2)
        k3 = _get_velocity(policy, x_k3, max(t_mid, 0.0), cond, local_cond, global_cond)
        
        # k4
        x_k4 = x_t + k3 * dt
        k4 = _get_velocity(policy, x_k4, max(t_next, 0.0), cond, local_cond, global_cond)
        
        # RK4 update
        x_t = x_t + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    if condition_mask is not None:
        x_t[condition_mask] = condition_data[condition_mask]
    
    return x_t


def detect_policy_type(policy) -> str:
    """检测策略类型"""
    class_name = policy.__class__.__name__
    
    if 'CFMTransformer' in class_name:
        return 'cfm_transformer'
    elif 'CFMUnet' in class_name or 'CFM' in class_name:
        return 'cfm_unet'
    elif 'Diffusion' in class_name:
        return 'diffusion'
    else:
        return 'unknown'


class InferenceTimeTracker:
    """追踪推理时间的包装器"""
    def __init__(self, policy):
        self.policy = policy
        self.inference_times = []
        self._original_predict_action = policy.predict_action
        
        # 包装 predict_action 方法
        def timed_predict_action(obs_dict, *args, **kwargs):
            start = time.perf_counter()
            result = self._original_predict_action(obs_dict, *args, **kwargs)
            end = time.perf_counter()
            self.inference_times.append(end - start)
            return result
        
        policy.predict_action = timed_predict_action
    
    def get_stats(self):
        """获取推理时间统计"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0,
                'std_inference_time': 0.0,
                'total_inference_calls': 0,
                'inference_calls_excluding_first': 0,
            }
        
        times = np.array(self.inference_times)
        # 排除第一次调用（通常包含 JIT 编译开销）
        times_excluding_first = times[1:] if len(times) > 1 else times
        
        return {
            'avg_inference_time': float(np.mean(times_excluding_first)) if len(times_excluding_first) > 0 else 0.0,
            'total_inference_time': float(np.sum(times)),
            'min_inference_time': float(np.min(times_excluding_first)) if len(times_excluding_first) > 0 else 0.0,
            'max_inference_time': float(np.max(times_excluding_first)) if len(times_excluding_first) > 0 else 0.0,
            'std_inference_time': float(np.std(times_excluding_first)) if len(times_excluding_first) > 0 else 0.0,
            'total_inference_calls': len(times),
            'inference_calls_excluding_first': len(times_excluding_first),
            'first_inference_time': float(times[0]) if len(times) > 0 else 0.0,
        }
    
    def restore(self):
        """恢复原始方法"""
        self.policy.predict_action = self._original_predict_action


def run_single_evaluation(policy, cfg, output_dir, steps, solver='euler'):
    """运行单次评估"""
    # 创建 env_runner
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    
    # 创建推理时间追踪器
    tracker = InferenceTimeTracker(policy)
    
    print(f"\n开始评估 (推理步数: {steps}, ODE solver: {solver})...")
    start_time = time.perf_counter()
    
    runner_log = env_runner.run(policy)
    
    end_time = time.perf_counter()
    total_wall_time = end_time - start_time
    
    # 获取推理时间统计
    inference_stats = tracker.get_stats()
    tracker.restore()  # 恢复原始方法
    
    # 添加评估元信息
    runner_log['eval/total_wall_time'] = total_wall_time
    runner_log['eval/inference_steps_used'] = steps
    runner_log['eval/ode_solver'] = solver
    
    # 添加推理时间统计
    runner_log['eval/avg_inference_time'] = inference_stats['avg_inference_time']
    runner_log['eval/total_inference_time'] = inference_stats['total_inference_time']
    runner_log['eval/min_inference_time'] = inference_stats['min_inference_time']
    runner_log['eval/max_inference_time'] = inference_stats['max_inference_time']
    runner_log['eval/std_inference_time'] = inference_stats['std_inference_time']
    runner_log['eval/total_inference_calls'] = inference_stats['total_inference_calls']
    runner_log['eval/inference_calls_excluding_first'] = inference_stats['inference_calls_excluding_first']
    runner_log['eval/first_inference_time'] = inference_stats['first_inference_time']
    
    return runner_log


def print_evaluation_summary(runner_log, checkpoint, steps, solver, policy_type):
    """打印评估摘要"""
    print(f"\n{'='*70}")
    print(f"CFM 评估完成! (策略类型: {policy_type})")
    print(f"{'='*70}")
    print(f"检查点: {checkpoint}")
    print(f"推理步数: {steps}")
    print(f"ODE Solver: {solver}")
    print(f"总评估时间: {runner_log['eval/total_wall_time']:.4f} 秒")
    
    # 测试得分
    if 'test/mean_score' in runner_log:
        print(f"\n测试得分:")
        print(f"  平均得分: {runner_log['test/mean_score']:.4f}")
    
    # 推理时间统计
    print(f"\n推理时间统计:")
    print(f"  总推理调用次数: {runner_log.get('eval/total_inference_calls', 0)}")
    print(f"  首次推理时间 (含JIT): {runner_log.get('eval/first_inference_time', 0):.5f} 秒")
    
    if runner_log.get('eval/inference_calls_excluding_first', 0) > 0:
        print(f"  平均推理时间 (排除首次): {runner_log['eval/avg_inference_time']:.5f} 秒")
        print(f"  最短推理时间: {runner_log.get('eval/min_inference_time', 0):.5f} 秒")
        print(f"  最长推理时间: {runner_log.get('eval/max_inference_time', 0):.5f} 秒")
        print(f"  推理时间标准差: {runner_log.get('eval/std_inference_time', 0):.5f} 秒")
        print(f"  总推理时间: {runner_log['eval/total_inference_time']:.5f} 秒")
        
        if runner_log['eval/total_wall_time'] > 0:
            inference_ratio = runner_log['eval/total_inference_time'] / runner_log['eval/total_wall_time']
            print(f"  推理时间占比: {inference_ratio*100:.1f}%")
            env_overhead = runner_log['eval/total_wall_time'] - runner_log['eval/total_inference_time']
            print(f"  环境模拟开销: {env_overhead:.4f} 秒 ({(1-inference_ratio)*100:.1f}%)")
    
    print(f"{'='*70}\n")


def save_results(runner_log, output_dir, checkpoint, steps, solver, policy_type):
    """保存评估结果"""
    # JSON 日志
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"评估日志已保存至: {out_path}")
    
    # CFM 性能报告
    perf_report_path = os.path.join(output_dir, 'cfm_performance_summary.txt')
    with open(perf_report_path, 'w', encoding='utf-8') as f:
        f.write("CFM (Conditional Flow Matching) 评估性能报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"策略类型: {policy_type}\n")
        f.write(f"检查点: {checkpoint}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"ODE Solver: {solver}\n")
        f.write(f"推理步数: {steps}\n")
        f.write(f"总评估耗时: {runner_log['eval/total_wall_time']:.4f} 秒\n\n")
        
        # 测试得分
        if 'test/mean_score' in json_log:
            f.write(f"测试得分:\n")
            f.write(f"  - 平均得分: {json_log['test/mean_score']}\n\n")
        
        # 推理时间
        if runner_log.get('eval/avg_inference_time', 0) > 0:
            f.write(f"推理时间统计:\n")
            f.write(f"  - 平均推理时间: {runner_log['eval/avg_inference_time']:.5f} 秒\n")
            f.write(f"  - 总推理时间: {runner_log['eval/total_inference_time']:.5f} 秒\n")
            f.write(f"  - 最短推理时间: {runner_log.get('eval/min_inference_time', 0):.5f} 秒\n")
            f.write(f"  - 最长推理时间: {runner_log.get('eval/max_inference_time', 0):.5f} 秒\n")
            
            if runner_log['eval/total_wall_time'] > 0:
                inference_ratio = runner_log['eval/total_inference_time'] / runner_log['eval/total_wall_time']
                f.write(f"  - 推理时间占比: {inference_ratio*100:.1f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nCFM 相比 Diffusion 的优势:\n")
        f.write("  1. 更少的推理步数 (通常 5-10 步 vs 50-100 步)\n")
        f.write("  2. 确定性 ODE 采样 (可选不同 solver)\n")
        f.write("  3. 更简单的训练目标 (直接回归速度场)\n")
    
    print(f"性能摘要已保存至: {perf_report_path}")


def run_step_comparison(policy, cfg, output_dir, checkpoint, step_list, solver, policy_type):
    """
    运行多个推理步数对比评估
    """
    comparison_results = []
    
    for steps in step_list:
        step_output_dir = os.path.join(output_dir, f'steps_{steps}')
        pathlib.Path(step_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 修改推理步数
        policy.num_inference_steps = steps
        
        # 运行评估
        runner_log = run_single_evaluation(policy, cfg, step_output_dir, steps, solver)
        
        # 收集结果
        result = {
            'steps': steps,
            'mean_score': runner_log.get('test/mean_score', 0),
            'wall_time': runner_log['eval/total_wall_time'],
            'avg_inference_time': runner_log.get('eval/avg_inference_time', 0),
            'total_inference_time': runner_log.get('eval/total_inference_time', 0),
            'total_inference_calls': runner_log.get('eval/total_inference_calls', 0),
            'first_inference_time': runner_log.get('eval/first_inference_time', 0),
        }
        comparison_results.append(result)
        
        # 保存单次结果
        save_results(runner_log, step_output_dir, checkpoint, steps, solver, policy_type)
    
    # 生成对比报告
    comparison_report_path = os.path.join(output_dir, 'step_comparison_report.txt')
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("CFM 推理步数对比报告\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"策略类型: {policy_type}\n")
        f.write(f"ODE Solver: {solver}\n")
        f.write(f"检查点: {checkpoint}\n\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'步数':<8} {'平均得分':<12} {'平均推理时间(ms)':<18} {'总推理时间(s)':<15} {'总耗时(s)':<12}\n")
        f.write("-" * 90 + "\n")
        
        for r in comparison_results:
            avg_time_ms = r['avg_inference_time'] * 1000
            f.write(f"{r['steps']:<8} {r['mean_score']:<12.4f} {avg_time_ms:<18.3f} {r['total_inference_time']:<15.3f} {r['wall_time']:<12.2f}\n")
        
        f.write("-" * 90 + "\n")
    
    print(f"\n推理步数对比报告已保存至: {comparison_report_path}")
    
    # 也保存 JSON 格式
    comparison_json_path = os.path.join(output_dir, 'step_comparison.json')
    json.dump(comparison_results, open(comparison_json_path, 'w'), indent=2)
    
    # 打印对比摘要
    print("\n" + "=" * 90)
    print("推理步数对比摘要")
    print("=" * 90)
    print(f"{'步数':<8} {'平均得分':<12} {'平均推理时间(ms)':<18} {'总推理时间(s)':<15} {'调用次数':<10}")
    print("-" * 90)
    for r in comparison_results:
        avg_time_ms = r['avg_inference_time'] * 1000
        print(f"{r['steps']:<8} {r['mean_score']:<12.4f} {avg_time_ms:<18.3f} {r['total_inference_time']:<15.3f} {r['total_inference_calls']:<10}")
    print("=" * 90 + "\n")


@click.command()
@click.option('-c', '--checkpoint', required=True, help='CFM 模型检查点路径')
@click.option('-o', '--output_dir', required=True, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='计算设备')
@click.option('--steps', default=None, type=int, help='推理步数 (默认使用训练配置)')
@click.option('--solver', default='euler', type=click.Choice(['euler', 'heun', 'midpoint', 'rk4']),
              help='ODE solver 类型')
@click.option('--compare-steps', default=None, type=str, 
              help='对比多个推理步数 (逗号分隔, 如: 1,5,10,20)')
@click.option('--use-ema/--no-ema', default=True, help='是否使用 EMA 模型')
def main(checkpoint, output_dir, device, steps, solver, compare_steps, use_ema):
    """
    CFM (Conditional Flow Matching) 专属评估脚本
    
    支持:
    - CFMUnetLowdimPolicy (U-Net + Flow Matching)
    - CFMTransformerLowdimPolicy (Transformer + Flow Matching)
    - 多种 ODE solver: euler, heun, midpoint, rk4
    - 推理步数对比实验
    """
    # 检查输出目录
    if os.path.exists(output_dir) and not compare_steps:
        click.confirm(f"输出路径 {output_dir} 已存在! 是否覆盖?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    print(f"正在加载检查点: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    
    # 创建 workspace
    try:
        workspace = cls(cfg, output_dir=output_dir)
    except TypeError:
        workspace = cls(cfg)
        try:
            workspace.output_dir = output_dir
        except AttributeError:
            pass
    
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取 policy
    policy = workspace.model
    if use_ema and cfg.training.use_ema:
        policy = workspace.ema_model
        print("使用 EMA 模型进行评估")
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # 检测策略类型
    policy_type = detect_policy_type(policy)
    print(f"检测到策略类型: {policy_type}")
    
    # 验证是否是 CFM policy
    if policy_type == 'diffusion':
        print("\n警告: 检测到 Diffusion Policy，此脚本专为 CFM 设计。")
        print("建议使用 eval.py 进行 Diffusion Policy 评估。")
        if not click.confirm("是否继续?"):
            return
    
    # 设置推理步数
    if steps is None:
        steps = getattr(policy, 'num_inference_steps', 10)
        print(f"使用默认推理步数: {steps}")
    else:
        policy.num_inference_steps = steps
        print(f"推理步数已设置为: {steps}")
    
    # 应用 ODE solver
    if solver != 'euler':
        print(f"应用 ODE solver: {solver}")
        patch_cfm_policy_solver(policy, solver, steps)
    
    # 运行评估
    if compare_steps:
        # 多步数对比
        step_list = [int(s.strip()) for s in compare_steps.split(',')]
        print(f"\n将进行推理步数对比: {step_list}")
        run_step_comparison(policy, cfg, output_dir, checkpoint, step_list, solver, policy_type)
    else:
        # 单次评估
        runner_log = run_single_evaluation(policy, cfg, output_dir, steps, solver)
        print_evaluation_summary(runner_log, checkpoint, steps, solver, policy_type)
        save_results(runner_log, output_dir, checkpoint, steps, solver, policy_type)


if __name__ == '__main__':
    main()
