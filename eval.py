"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output --steps 50
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
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# ================= 导入 DDIM 调度器 =================
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# ==========================================================

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--steps', default=None, type=int, help="Override inference steps (e.g. 50)")
def main(checkpoint, output_dir, device, steps):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

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

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # ================= 修改 Scheduler 的逻辑 =================
    if steps is not None and steps != getattr(policy, 'num_inference_steps', steps):
        if hasattr(policy, 'num_inference_steps') and hasattr(policy, 'noise_scheduler'):
            print(f"Overriding inference steps to {steps} and switching to DDIMScheduler!")
            policy.num_inference_steps = steps
            policy.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
        else:
            print("Warning: policy has no diffusion scheduler; --steps override skipped (likely VAE backend).")
    # ===================================================================
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    
    print("Starting evaluation...")
    start_time = time.perf_counter()
    
    runner_log = env_runner.run(policy)
    
    end_time = time.perf_counter()
    total_wall_time = end_time - start_time
    # ==========================================================
    
    # ================= 确保必要的时间信息添加到日志 =================
    # 添加总评估时间
    runner_log['eval/total_wall_time'] = total_wall_time
    runner_log['eval/inference_steps_used'] = getattr(policy, 'num_inference_steps', steps)
    
    # 检查env_runner是否已经记录了推理时间，如果没有则添加占位符
    if 'eval/avg_inference_time' not in runner_log:
        runner_log['eval/avg_inference_time'] = 0.0
        runner_log['eval/total_inference_time'] = 0.0
        runner_log['eval/inference_calls_excluding_first'] = 0
        runner_log['eval/total_inference_calls'] = 0
    
    # 在控制台打印时间信息
    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"总评估时间: {total_wall_time:.4f} 秒")
    print(f"推理步数: {runner_log['eval/inference_steps_used']}")
    
    # 如果有推理时间统计，也打印出来
    if runner_log.get('eval/avg_inference_time', 0) > 0:
        print(f"纯模型平均推理时间: {runner_log['eval/avg_inference_time']:.5f} 秒")
        print(f"纯模型总推理时间: {runner_log['eval/total_inference_time']:.5f} 秒")
        print(f"推理调用次数(排除第一次): {runner_log.get('eval/inference_calls_excluding_first', 0)}")
        print(f"总推理调用次数: {runner_log.get('eval/total_inference_calls', 0)}")
        
        # 计算推理时间占比
        if runner_log['eval/total_inference_time'] > 0:
            inference_ratio = runner_log['eval/total_inference_time'] / total_wall_time
            print(f"推理时间占比: {inference_ratio*100:.1f}%")
            runner_log['eval/inference_time_ratio'] = inference_ratio
            
            # 环境模拟开销
            env_overhead = total_wall_time - runner_log['eval/total_inference_time']
            print(f"环境模拟开销: {env_overhead:.4f} 秒")
            runner_log['eval/env_sim_overhead'] = env_overhead
    print(f"{'='*60}")
    # ===============================================================
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    
    # ================= 生成性能报告 =================
    perf_report_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(perf_report_path, 'w') as f:
        f.write("Diffusion Policy 评估性能报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"检查点: {checkpoint}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"设备: {device}\n")
        f.write(f"推理步数: {runner_log['eval/inference_steps_used']}\n")
        f.write(f"总评估耗时: {total_wall_time:.4f} 秒\n")
        
        # 如果有推理时间统计，写入报告
        if 'eval/avg_inference_time' in runner_log and runner_log['eval/avg_inference_time'] > 0:
            f.write(f"\n纯模型推理时间统计:\n")
            f.write(f"  - 平均推理时间: {runner_log['eval/avg_inference_time']:.5f} 秒\n")
            f.write(f"  - 总推理时间: {runner_log['eval/total_inference_time']:.5f} 秒\n")
            f.write(f"  - 最短推理时间: {runner_log.get('eval/min_inference_time', 0):.5f} 秒\n")
            f.write(f"  - 最长推理时间: {runner_log.get('eval/max_inference_time', 0):.5f} 秒\n")
            f.write(f"  - 推理时间标准差: {runner_log.get('eval/std_inference_time', 0):.5f} 秒\n")
            f.write(f"  - 推理调用次数(排除第一次): {runner_log.get('eval/inference_calls_excluding_first', 0)}\n")
            f.write(f"  - 总推理调用次数: {runner_log.get('eval/total_inference_calls', 0)}\n")
            
            # 推理时间占比
            if 'eval/inference_time_ratio' in runner_log:
                f.write(f"  - 推理时间占比: {runner_log['eval/inference_time_ratio']*100:.1f}%\n")
            
            # 环境模拟开销
            if 'eval/env_sim_overhead' in runner_log:
                f.write(f"  - 环境模拟开销: {runner_log['eval/env_sim_overhead']:.4f} 秒\n")
        
        # 测试得分
        if 'test/mean_score' in json_log:
            f.write(f"\n测试得分:\n")
            f.write(f"  - 平均得分: {json_log['test/mean_score']}\n")
        
        # 统计episode数量
        test_episodes = len([k for k in json_log.keys() if k.startswith('test/sim_max_reward_')])
        train_episodes = len([k for k in json_log.keys() if k.startswith('train/sim_max_reward_')])
        f.write(f"\nEpisode 统计:\n")
        f.write(f"  - 测试集episode数量: {test_episodes}\n")
        f.write(f"  - 训练集episode数量: {train_episodes}\n")
        
        # 如果有估算的episode数量，也写入
        if runner_log.get('eval/total_inference_calls', 0) > 0:
            try:
                # 从配置中获取episode步数，默认为300（Pusht环境）
                steps_per_episode = cfg.task.env_runner.steps_per_episode
            except AttributeError:
                steps_per_episode = 300
            
            estimated_episodes = runner_log['eval/total_inference_calls'] / steps_per_episode
            f.write(f"  - 估计总episode数量: {estimated_episodes:.1f}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"性能摘要已保存至: {perf_report_path}")

if __name__ == '__main__':
    main()