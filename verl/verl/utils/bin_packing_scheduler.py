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
Bin Packing调度器：根据avg response length将prompts分配到replicas
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class BinPackingScheduler:
    """Bin Packing调度器，用于将prompts根据avg response length分配到replicas"""
    
    def __init__(self, schedule_file: Optional[str] = None):
        """
        初始化调度器
        
        Args:
            schedule_file: 调度结果JSON文件路径。如果为None，尝试从环境变量读取
        """
        if schedule_file is None:
            schedule_file = os.getenv("VERL_BIN_PACKING_SCHEDULE_FILE", None)
        
        if schedule_file is None:
            raise ValueError(
                "schedule_file must be provided or set via VERL_BIN_PACKING_SCHEDULE_FILE environment variable"
            )
        
        self.schedule_file = schedule_file
        self.schedule = self._load_schedule()
        self._validate_schedule()
    
    def _load_schedule(self) -> dict:
        """加载调度结果"""
        with open(self.schedule_file, 'r') as f:
            return json.load(f)
    
    def _validate_schedule(self):
        """验证调度结果的完整性"""
        assert 'replicas' in self.schedule, "schedule must contain 'replicas'"
        assert 'num_replicas' in self.schedule, "schedule must contain 'num_replicas'"
        
        # 验证所有prompts都被分配
        all_prompt_ids = set()
        for replica in self.schedule['replicas']:
            assert 'prompt_ids' in replica, f"replica {replica.get('replica_id')} must contain 'prompt_ids'"
            prompt_ids = set(replica['prompt_ids'])
            assert len(all_prompt_ids & prompt_ids) == 0, f"Duplicate prompt IDs found: {all_prompt_ids & prompt_ids}"
            all_prompt_ids.update(prompt_ids)
        
        expected_prompts = self.schedule.get('num_prompts', len(all_prompt_ids))
        assert len(all_prompt_ids) == expected_prompts, (
            f"Expected {expected_prompts} prompts, but found {len(all_prompt_ids)}"
        )
    
    def get_replica_for_prompt(self, prompt_id: int) -> int:
        """
        获取指定prompt应该被分配到哪个replica
        
        Args:
            prompt_id: Prompt的ID（原始prompt索引，从0开始）
        
        Returns:
            replica_id: Replica的ID
        """
        for replica in self.schedule['replicas']:
            if prompt_id in replica['prompt_ids']:
                return replica['replica_id']
        
        raise ValueError(f"Prompt {prompt_id} not found in schedule")
    
    def get_prompts_for_replica(self, replica_id: int) -> List[int]:
        """
        获取指定replica应该处理的所有prompt IDs
        
        Args:
            replica_id: Replica的ID
        
        Returns:
            List of prompt IDs
        """
        for replica in self.schedule['replicas']:
            if replica['replica_id'] == replica_id:
                return replica['prompt_ids']
        
        raise ValueError(f"Replica {replica_id} not found in schedule")
    
    def create_prompt_to_replica_mapping(self) -> Dict[int, int]:
        """
        创建prompt_id到replica_id的映射
        
        Returns:
            Dict mapping prompt_id -> replica_id
        """
        mapping = {}
        for replica in self.schedule['replicas']:
            replica_id = replica['replica_id']
            for prompt_id in replica['prompt_ids']:
                mapping[prompt_id] = replica_id
        return mapping
    
    def get_replica_order(self) -> List[int]:
        """
        获取replica的处理顺序（按replica_id排序）
        
        Returns:
            List of replica IDs in order
        """
        return sorted([replica['replica_id'] for replica in self.schedule['replicas']])


def get_bin_packing_scheduler() -> Optional[BinPackingScheduler]:
    """
    获取bin packing调度器（如果配置了）
    
    Returns:
        BinPackingScheduler实例，如果未配置则返回None
    """
    schedule_file = os.getenv("VERL_BIN_PACKING_SCHEDULE_FILE", None)
    if schedule_file is None:
        return None
    
    try:
        return BinPackingScheduler(schedule_file)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load bin packing schedule: {e}")
        return None


