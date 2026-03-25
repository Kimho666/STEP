"""
Action Predictor Dataset
为Action Predictor训练提供适当的数据格式
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class ActionPredictorPushTDataset(BaseLowdimDataset):
    """
    为Action Predictor设计的PushT数据集
    
    除了标准的obs和action外，还提供prev_action（上一个action chunk的专家动作）
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        prev_action_horizon: int = 8,  # 上一个action chunk的长度
        obs_key: str = 'keypoint',
        state_key: str = 'state',
        action_key: str = 'action',
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
    ):
        super().__init__()
        
        # 加载数据
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])
        
        # 划分训练/验证集
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        # 创建采样器
        # 需要额外的pad_before来获取prev_action
        extended_pad_before = pad_before + prev_action_horizon
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon + prev_action_horizon,  # 扩展序列长度
            pad_before=extended_pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.prev_action_horizon = prev_action_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.extended_pad_before = extended_pad_before
    
    def get_validation_dataset(self):
        """获取验证数据集"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon + self.prev_action_horizon,
            pad_before=self.extended_pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """获取归一化器"""
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        
        # 只对obs和action进行归一化
        normalizer.fit(
            data={'obs': data['obs'], 'action': data['action']}, 
            last_n_dims=1, 
            mode=mode, 
            **kwargs
        )
        
        # prev_action使用与action相同的归一化参数
        normalizer['prev_action'] = normalizer['action']
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        """将采样转换为数据字典"""
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:, :2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)
        
        data = {
            'obs': obs,
            'action': sample[self.action_key],
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.sampler.sample_sequence(idx)
        
        # 处理keypoint和state
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:, :2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)
        
        action = sample[self.action_key]
        
        # 分割prev_action和当前action
        # 假设采样的序列是 [prev_action_horizon + horizon] 长度
        total_len = action.shape[0]
        
        # prev_action是序列的前prev_action_horizon步
        prev_action = action[:self.prev_action_horizon]
        
        # 当前的obs和action是从prev_action_horizon开始
        current_obs = obs[self.prev_action_horizon:]
        current_action = action[self.prev_action_horizon:]
        
        data = {
            'obs': current_obs,  # [horizon, obs_dim]
            'action': current_action,  # [horizon, action_dim]  
            'prev_action': prev_action,  # [prev_action_horizon, action_dim]
        }
        
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class ActionPredictorDatasetWrapper(BaseLowdimDataset):
    """
    通用数据集包装器
    将任意BaseLowdimDataset包装为支持prev_action的格式
    """
    
    def __init__(
        self,
        base_dataset: BaseLowdimDataset,
        prev_action_horizon: int = 8,
        use_zero_prev_for_start: bool = True,  # 序列开始时是否使用零作为prev_action
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.prev_action_horizon = prev_action_horizon
        self.use_zero_prev_for_start = use_zero_prev_for_start
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        
        action = sample['action']  # [horizon, action_dim]
        horizon, action_dim = action.shape
        
        # 创建prev_action
        # 方案1：使用当前序列的前几步作为prev_action（模拟之前chunk的结尾）
        # 方案2：使用零初始化（表示episode开始）
        
        if horizon >= self.prev_action_horizon and not self.use_zero_prev_for_start:
            # 使用序列开头作为prev_action
            prev_action = action[:self.prev_action_horizon].clone()
        else:
            # 使用零初始化
            prev_action = torch.zeros(
                self.prev_action_horizon, action_dim, 
                dtype=action.dtype
            )
        
        sample['prev_action'] = prev_action
        return sample
    
    def get_normalizer(self, **kwargs):
        normalizer = self.base_dataset.get_normalizer(**kwargs)
        # prev_action使用与action相同的归一化
        normalizer['prev_action'] = normalizer['action']
        return normalizer
    
    def get_validation_dataset(self):
        val_base = self.base_dataset.get_validation_dataset()
        return ActionPredictorDatasetWrapper(
            val_base,
            self.prev_action_horizon,
            self.use_zero_prev_for_start
        )
    
    def get_all_actions(self) -> torch.Tensor:
        return self.base_dataset.get_all_actions()
