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
Bin Packing调度：根据avg response length将prompts分配到replicas
"""

import os
from typing import Optional

from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.base.worker_group import WorkerGroup
from verl.utils.bin_packing_scheduler import BinPackingScheduler, get_bin_packing_scheduler


def dispatch_bin_packing_data_proto(worker_group: WorkerGroup, *args, **kwargs):
    """
    基于Bin Packing调度结果的DataProto分发函数
    
    这个函数会根据bin packing调度结果，将prompts重新排列并分配到对应的replicas。
    考虑TP（Tensor Parallelism），多个workers组成一个replica。
    
    Args:
        worker_group: WorkerGroup实例
        *args: DataProto参数
        **kwargs: DataProto关键字参数
    
    Returns:
        splitted_args, splitted_kwargs: 切分后的参数
    """
    # 获取bin packing调度器
    scheduler = get_bin_packing_scheduler()
    if scheduler is None:
        # 如果未配置调度器，回退到默认的DP分发
        from verl.single_controller.base.decorator import dispatch_dp_compute_data_proto
        return dispatch_dp_compute_data_proto(worker_group, *args, **kwargs)
    
    # 获取TP size（从worker_group或config中）
    tp_size = getattr(worker_group, '_tp_size', None)
    if tp_size is None:
        # 尝试从环境变量或配置中获取
        tp_size = int(os.getenv("VERL_TP_SIZE", "1"))
    
    num_workers = worker_group.world_size
    num_replicas = num_workers // tp_size if tp_size > 0 else num_workers
    
    # 验证replicas数量
    expected_replicas = scheduler.schedule.get('num_replicas', num_replicas)
    if expected_replicas != num_replicas:
        import warnings
        warnings.warn(
            f"Schedule expects {expected_replicas} replicas, but worker_group has {num_replicas} replicas "
            f"(world_size={num_workers}, tp_size={tp_size}). Falling back to default dispatch."
        )
        from verl.single_controller.base.decorator import dispatch_dp_compute_data_proto
        return dispatch_dp_compute_data_proto(worker_group, *args, **kwargs)
    
    # 获取rollout.n（每个prompt的重复次数）
    rollout_n = int(os.getenv("VERL_ROLLOUT_N", "1"))
    # 尝试从meta_info中获取
    if len(args) > 0 and isinstance(args[0], DataProto):
        rollout_n = args[0].meta_info.get("rollout_n", rollout_n)
    
    # 处理每个DataProto参数
    splitted_args = []
    for arg in args:
        if isinstance(arg, (DataProto, DataProtoFuture)):
            splitted_arg = _dispatch_bin_packing_single_proto(
                arg, scheduler, num_replicas, tp_size, rollout_n, worker_group
            )
            splitted_args.append(splitted_arg)
        else:
            # 非DataProto参数，按replica数量复制
            splitted_args.append([arg] * num_replicas)
    
    # 处理关键字参数
    splitted_kwargs = {}
    for key, val in kwargs.items():
        if isinstance(val, (DataProto, DataProtoFuture)):
            splitted_kwargs[key] = _dispatch_bin_packing_single_proto(
                val, scheduler, num_replicas, tp_size, rollout_n, worker_group
            )
        else:
            splitted_kwargs[key] = [val] * num_replicas
    
    return tuple(splitted_args), splitted_kwargs


def _dispatch_bin_packing_single_proto(
    data_proto: DataProto | DataProtoFuture,
    scheduler: BinPackingScheduler,
    num_replicas: int,
    tp_size: int,
    rollout_n: int,
    worker_group: WorkerGroup,
) -> list[DataProto | DataProtoFuture]:
    """
    对单个DataProto进行bin packing调度分发
    
    Args:
        data_proto: 要分发的DataProto
        scheduler: BinPackingScheduler实例
        num_replicas: Replicas数量
        tp_size: Tensor Parallelism大小
        rollout_n: 每个prompt的重复次数
        worker_group: WorkerGroup实例
    
    Returns:
        List of DataProto，每个对应一个worker（但相同replica的workers会收到相同数据）
    """
    # 如果是DataProtoFuture，需要先获取
    if isinstance(data_proto, DataProtoFuture):
        # 对于Future，我们需要创建一个新的dispatch函数
        # 这里简化处理，假设Future会在后续处理
        import warnings
        warnings.warn("DataProtoFuture not fully supported in bin packing dispatch, using default chunk")
        return data_proto.chunk(worker_group.world_size)
    
    data_len = len(data_proto)
    
    # 计算原始prompts数量
    num_original_prompts = data_len // rollout_n if rollout_n > 0 else data_len
    
    # 创建prompt_id到replica_id的映射
    prompt_to_replica = scheduler.create_prompt_to_replica_mapping()
    
    # 创建重新排列的索引
    # 目标：将数据重新排列，使得每个replica的prompts（包括所有n个副本）连续
    reorder_indices = []
    replica_prompt_counts = [0] * num_replicas
    
    # 按replica顺序收集prompts
    for replica_id in range(num_replicas):
        prompt_ids = scheduler.get_prompts_for_replica(replica_id)
        for prompt_id in prompt_ids:
            if prompt_id < num_original_prompts:
                # 这个prompt的所有n个副本的索引
                start_idx = prompt_id * rollout_n
                end_idx = start_idx + rollout_n
                for idx in range(start_idx, end_idx):
                    reorder_indices.append(idx)
                replica_prompt_counts[replica_id] += rollout_n
    
    # 验证索引数量
    if len(reorder_indices) != data_len:
        import warnings
        warnings.warn(
            f"Bin packing reorder indices count ({len(reorder_indices)}) != data length ({data_len}). "
            f"num_original_prompts={num_original_prompts}, rollout_n={rollout_n}. "
            f"Using default chunk dispatch."
        )
        return data_proto.chunk(worker_group.world_size)
    
    # 重新排列数据
    reordered_data = data_proto.select_idxs(reorder_indices)
    
    # 按照replica切分数据
    replica_data_list = []
    current_idx = 0
    for replica_id in range(num_replicas):
        replica_size = replica_prompt_counts[replica_id]
        if replica_size > 0:
            replica_data = reordered_data.slice(current_idx, current_idx + replica_size)
            replica_data_list.append(replica_data)
            current_idx += replica_size
        else:
            # 空replica，创建空的DataProto
            empty_data = data_proto.slice(0, 0)  # 创建空DataProto
            replica_data_list.append(empty_data)
    
    # 现在需要将replica数据分配给workers
    # 由于TP=2，每个replica有2个workers，需要将replica数据复制给这2个workers
    worker_data_list = []
    for worker_rank in range(worker_group.world_size):
        replica_id = worker_rank // tp_size
        if replica_id < len(replica_data_list):
            worker_data_list.append(replica_data_list[replica_id])
        else:
            # 超出replica范围，创建空DataProto
            empty_data = data_proto.slice(0, 0)
            worker_data_list.append(empty_data)
    
    return worker_data_list

