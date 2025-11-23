# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shuffle调度器：完全随机打乱所有prompts
"""

import os
import random
from typing import Optional


class ShuffleScheduler:
    """Shuffle调度器，用于完全随机打乱所有prompts"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化调度器
        
        Args:
            seed: 随机种子，如果为None，使用环境变量VERL_SHUFFLE_SEED或系统时间
        """
        if seed is None:
            seed = os.getenv("VERL_SHUFFLE_SEED", None)
            if seed is not None:
                seed = int(seed)
            else:
                seed = None  # 使用系统时间作为种子
        
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def get_shuffled_indices(self, data_len: int, rollout_n: int) -> list[int]:
        """
        获取shuffle后的索引
        
        Args:
            data_len: 总数据长度（repeat后的长度）
            rollout_n: 每个prompt的重复次数
        
        Returns:
            List of shuffled indices
        """
        num_original_prompts = data_len // rollout_n if rollout_n > 0 else data_len
        
        # 创建原始prompt的索引列表
        original_prompt_indices = list(range(num_original_prompts))
        
        # 随机打乱原始prompt索引
        shuffled_prompt_indices = original_prompt_indices.copy()
        random.shuffle(shuffled_prompt_indices)
        
        # 为每个prompt创建所有n个副本的索引
        shuffled_indices = []
        for prompt_id in shuffled_prompt_indices:
            start_idx = prompt_id * rollout_n
            end_idx = start_idx + rollout_n
            # 保持同一个prompt的n个副本连续
            for idx in range(start_idx, end_idx):
                shuffled_indices.append(idx)
        
        return shuffled_indices
    
    def get_shuffled_indices_full(self, data_len: int) -> list[int]:
        """
        完全随机打乱所有数据（包括同一个prompt的多个副本）
        
        Args:
            data_len: 总数据长度
        
        Returns:
            List of fully shuffled indices
        """
        indices = list(range(data_len))
        random.shuffle(indices)
        return indices


def get_shuffle_scheduler() -> Optional[ShuffleScheduler]:
    """
    获取shuffle调度器（如果配置了）
    
    Returns:
        ShuffleScheduler实例，如果未配置则返回None
    """
    shuffle_enabled = os.getenv("VERL_ENABLE_SHUFFLE", "false").lower() in ["true", "1", "yes"]
    if not shuffle_enabled:
        return None
    
    try:
        seed = os.getenv("VERL_SHUFFLE_SEED", None)
        if seed is not None:
            seed = int(seed)
        return ShuffleScheduler(seed=seed)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to create shuffle scheduler: {e}")
        return None


