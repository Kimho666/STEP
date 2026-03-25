"""
Action Predictor Training Workspace
用于独立训练Action Predictor模型
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.policy.action_predictor_lowdim_policy import (
    ActionPredictorLowdimPolicy,
    ActionPredictorDatasetWrapper,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainActionPredictorLowdimWorkspace(BaseWorkspace):
    """
    Action Predictor训练工作区
    支持 transformer 与 VAE 两种后端
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        # allow adding fields like cfg.vae_model when struct configs are enforced
        OmegaConf.set_struct(cfg, False)
        super().__init__(cfg, output_dir=output_dir)

        # 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 配置模型：支持 transformer 与 VAE 两种后端
        policy_cfg = copy.deepcopy(cfg.policy)
        # allow temporary mutation for backend switch
        OmegaConf.set_struct(policy_cfg, False)
        
        vae_model = None
        if cfg.get('use_vae', False):
            # instantiate VAE explicitly so the policy sees a real module, not a DictConfig
            vae_model = hydra.utils.instantiate(cfg.vae_model)
            policy_cfg.backend = 'vae'
        OmegaConf.set_struct(policy_cfg, True)

        self.model: ActionPredictorLowdimPolicy
        # do not re-instantiate nested model when we've already built it
        if vae_model is not None:
            instantiate_kwargs = {'_recursive_': False, 'model': vae_model}
        else:
            instantiate_kwargs = {}
        self.model = hydra.utils.instantiate(policy_cfg, **instantiate_kwargs)

        # 配置优化器
        # 直接使用 torch.optim.AdamW，避免 hydra 实例化的问题
        optimizer_cfg = cfg.optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_cfg.lr,
            betas=tuple(optimizer_cfg.betas),
            eps=optimizer_cfg.eps,
            weight_decay=optimizer_cfg.weight_decay
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # 恢复训练
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # 配置数据集
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        
        # 包装数据集以添加prev_action
        wrapped_dataset = ActionPredictorDatasetWrapper(
            base_dataset=dataset,
            prev_action_horizon=cfg.prev_action_horizon,
            n_action_steps=cfg.n_action_steps,
            pad_before=cfg.task.dataset.get('pad_before', 0)
        )
        
        train_dataloader = DataLoader(wrapped_dataset, **cfg.dataloader)
        normalizer = wrapped_dataset.get_normalizer()

        # 配置验证数据集
        val_dataset = wrapped_dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # 配置学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # 配置环境运行器（可选）
        env_runner: BaseLowdimRunner = None
        if cfg.task.get('env_runner') is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir
            )

        # 配置日志
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # 配置checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # 设备转移
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # 保存batch用于采样
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # 训练循环
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # ========= 训练 ==========
                train_losses = list()
                self.model.train()
                
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # 设备转移
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # 计算损失
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # 优化器步骤
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # 日志
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # epoch结束，更新train_loss为平均值
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= 验证 ==========
                self.model.eval()

                # 运行rollout（如果配置了环境运行器）
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    step_log.update(runner_log)

                # 运行验证
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # 采样评估
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        prev_action = batch['prev_action']
                        
                        result = self.model.predict_action(obs_dict, prev_action=prev_action)
                        pred_action = result['action_pred']
                        
                        # 计算MSE
                        target_action = gt_action[:, :pred_action.shape[1]]
                        mse = torch.nn.functional.mse_loss(pred_action, target_action)
                        step_log['train_action_mse_error'] = mse.item()
                        
                        del batch, obs_dict, gt_action, result, pred_action, mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # 日志记录
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainActionPredictorLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
