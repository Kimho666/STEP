"""
Training workspace for image-based Action Predictor (lightweight).
Derived from lowdim workspace but removes BaseLowdimDataset assertion.
"""
import os
import pathlib
import copy
import random
import shutil
import tqdm
import hydra
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.action_predictor_image_policy import ActionPredictorImagePolicy
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorDatasetWrapper
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainActionPredictorImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: ActionPredictorImagePolicy = hydra.utils.instantiate(cfg.policy)

        opt_cfg = cfg.optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay
        )
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            last_ckpt = self.get_checkpoint_path()
            if last_ckpt.is_file():
                print(f"Resuming from checkpoint {last_ckpt}")
                self.load_checkpoint(path=last_ckpt)

        # dataset（图像版，不做类型断言）
        base_dataset = hydra.utils.instantiate(cfg.task.dataset)
        wrapped_dataset = ActionPredictorDatasetWrapper(
            base_dataset=base_dataset,
            prev_action_horizon=cfg.prev_action_horizon,
            n_action_steps=cfg.n_action_steps,
            pad_before=cfg.task.dataset.get('pad_before', 0)
        )
        train_dataloader = DataLoader(wrapped_dataset, **cfg.dataloader)
        normalizer = wrapped_dataset.get_normalizer()

        val_dataset = wrapped_dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        env_runner = None
        if cfg.task.get('env_runner') is not None:
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                self.model.train()

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        self.global_step += 1

                        if cfg.training.max_train_steps and batch_idx >= cfg.training.max_train_steps:
                            break

                # ========== 验证 ==========
                self.model.eval()
                val_losses = list()
                with torch.no_grad(), tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                    for batch_idx, batch in enumerate(vepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        val_loss = self.model.compute_loss(batch).item()
                        val_losses.append(val_loss)
                        vepoch.set_postfix(loss=val_loss, refresh=False)
                        if cfg.training.max_val_steps and batch_idx >= cfg.training.max_val_steps:
                            break

                train_loss = float(np.mean(train_losses))
                val_loss = float(np.mean(val_losses))
                epoch_log = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'global_step': self.global_step,
                    'epoch': self.epoch
                }

                wandb.log(epoch_log, step=self.global_step)
                json_logger.log(epoch_log)

                # checkpoint
                ckpt_path = self.save_checkpoint()
                metric_dict = {
                    'val_loss': val_loss,
                    'epoch': self.epoch
                }
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None and topk_ckpt_path != ckpt_path:
                    shutil.copy(ckpt_path, topk_ckpt_path)
                if cfg.checkpoint.save_last_ckpt:
                    last_ckpt_path = os.path.join(self.output_dir, 'checkpoints', 'latest.ckpt')
                    if os.path.abspath(ckpt_path) != os.path.abspath(last_ckpt_path):
                        shutil.copy(ckpt_path, last_ckpt_path)

                self.epoch += 1

                if cfg.training.max_train_steps and self.global_step >= cfg.training.max_train_steps:
                    break

        if env_runner is not None and train_sampling_batch is not None:
            self.model.eval()
            env_runner.run(self.model)

        if wandb_run is not None:
            wandb_run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    hydra.main(config_path=str(pathlib.Path(__file__).parent.parent.parent/'config'), config_name=args.config, version_base=None)(lambda cfg: TrainActionPredictorImageWorkspace(cfg))


if __name__ == "__main__":
    main()
