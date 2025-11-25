#!/usr/bin/env python3
"""
重新计算default模式的token统计
通过每个prompt的avg response length反推每个worker生产的response token数
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def recalculate_default_tokens(metrics_file: str, output_file: str = None):
    """
    重新计算default模式的token统计
    
    Args:
        metrics_file: 输入的metrics JSON文件路径
        output_file: 输出的JSON文件路径（如果为None，则覆盖原文件）
    """
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    num_workers = summary.get('num_workers', 16)
    num_replicas = summary.get('num_replicas', 8)
    tp_size = summary.get('tp_size', 2)
    
    # 获取per_prompt_avg_response_lengths
    per_batch_metrics = data.get('per_batch_metrics', [])
    if not per_batch_metrics:
        print(f"Error: No per_batch_metrics found in {metrics_file}")
        return False
    
    # 从第一个batch获取per_prompt_avg_response_lengths
    first_batch = per_batch_metrics[0]
    per_prompt_avg_response_lengths = first_batch.get('per_prompt_avg_response_lengths', {})
    
    if not per_prompt_avg_response_lengths:
        print(f"Error: No per_prompt_avg_response_lengths found in {metrics_file}")
        return False
    
    # 获取actual_n (val_kwargs.n)
    # 在validation中，val_kwargs_n=1，所以每个prompt只生成1次
    # per_prompt_avg_response_lengths已经是单次生成的响应长度
    val_kwargs_n = first_batch.get('val_kwargs_n', 1)
    rollout_n = first_batch.get('rollout_n', 1)  # 用于参考
    num_original_prompts = first_batch.get('num_original_prompts', len(per_prompt_avg_response_lengths))
    
    print(f"Processing {metrics_file}")
    print(f"  num_workers: {num_workers}, num_replicas: {num_replicas}, tp_size: {tp_size}")
    print(f"  num_original_prompts: {num_original_prompts}, val_kwargs_n: {val_kwargs_n}, rollout_n: {rollout_n}")
    print(f"  Found {len(per_prompt_avg_response_lengths)} prompts with avg response lengths")
    
    # 计算每个replica的token数（default模式：顺序分配）
    prompts_per_replica = num_original_prompts // num_replicas
    remainder = num_original_prompts % num_replicas
    
    per_replica_tokens = {}
    prompt_idx = 0
    
    for replica_id in range(num_replicas):
        # 计算这个replica分到的prompt数量
        num_prompts = prompts_per_replica + (1 if replica_id < remainder else 0)
        replica_tokens = 0
        
        for _ in range(num_prompts):
            if prompt_idx < num_original_prompts:
                # 使用avg_response_length来计算总token数
                # 在validation中，val_kwargs_n=1，所以avg_response_length就是实际响应长度
                # per_prompt_avg_response_lengths中的值已经是单次生成的响应长度
                prompt_id_str = str(prompt_idx)
                if prompt_id_str in per_prompt_avg_response_lengths:
                    avg_length = per_prompt_avg_response_lengths[prompt_id_str]
                    # 直接使用avg_response_length（因为val_kwargs_n=1，所以avg = 实际值）
                    replica_tokens += avg_length
                prompt_idx += 1
        
        per_replica_tokens[replica_id] = replica_tokens
        print(f"  Replica {replica_id}: {num_prompts} prompts, {replica_tokens:.0f} tokens")
    
    # 更新per_worker_summary
    per_worker_summary = data.get('per_worker_summary', [])
    total_recalculated = sum(per_replica_tokens.values())  # 直接使用replica tokens的总和，避免重复计算
    
    for worker in per_worker_summary:
        worker_rank = worker.get('worker_rank', 0)
        replica_id = worker.get('replica_id', worker_rank // tp_size)
        
        old_tokens = worker.get('total_tokens', 0)
        
        if replica_id in per_replica_tokens:
            new_tokens = int(per_replica_tokens[replica_id])
            worker['total_tokens'] = new_tokens
            worker['avg_tokens_per_batch'] = float(new_tokens) / worker.get('num_batches', 1)
        else:
            print(f"  Warning: Replica {replica_id} not found in per_replica_tokens")
    
    # 更新per_batch_metrics中的all_workers
    for batch_idx, batch_metrics in enumerate(per_batch_metrics):
        all_workers = batch_metrics.get('all_workers', [])
        
        for worker_data in all_workers:
            worker_rank = worker_data.get('worker_rank', 0)
            replica_id = worker_data.get('replica_id', worker_rank // tp_size)
            
            if replica_id in per_replica_tokens:
                new_tokens = int(per_replica_tokens[replica_id])
                worker_data['response_tokens'] = new_tokens
            else:
                print(f"  Warning: Replica {replica_id} not found in per_batch_metrics[{batch_idx}]")
        
        # 更新all_replicas
        all_replicas = batch_metrics.get('all_replicas', [])
        for replica_data in all_replicas:
            replica_id = replica_data.get('replica_id', 0)
            if replica_id in per_replica_tokens:
                new_tokens = int(per_replica_tokens[replica_id])
                replica_data['response_tokens'] = new_tokens
    
    # 验证总和
    total_expected = summary.get('total_response_tokens', 0)
    print(f"\n  Total recalculated tokens: {total_recalculated:.0f}")
    print(f"  Total expected tokens: {total_expected:.0f}")
    print(f"  Difference: {abs(total_recalculated - total_expected):.0f}")
    
    if abs(total_recalculated - total_expected) > 100:  # 允许100 tokens的误差
        print(f"  Warning: Significant difference between recalculated and expected tokens!")
    
    # 保存更新后的数据
    if output_file is None:
        output_file = metrics_file
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  Updated metrics saved to: {output_file}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python recalculate_default_tokens.py <metrics_file> [output_file]")
        print("Example: python recalculate_default_tokens.py logs/.../rollout_worker_metrics_default_*.json")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(metrics_file).exists():
        print(f"Error: File not found: {metrics_file}")
        sys.exit(1)
    
    success = recalculate_default_tokens(metrics_file, output_file)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()

