#!/usr/bin/env python3
"""
Bin Packing调度算法
将prompts根据(prompt_len + avg_response_len)的估计总tokens分配到replicas，实现负载均衡
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple


def best_fit_decreasing(items: List[Tuple[int, float]], num_bins: int) -> Tuple[List[List[Tuple[int, float]]], List[float]]:
    """
    Best Fit Decreasing算法：将items分配到bins中，使得负载最均衡
    
    Args:
        items: List of (prompt_id, weight) tuples, where weight是估计的prompt+response总tokens
        num_bins: Number of bins (replicas)
    
    Returns:
        bins: List of lists, each containing (prompt_id, weight) tuples
        bin_sums: List of total weights for each bin
    """
    # 按weight降序排序
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    
    # 初始化bins
    bins = [[] for _ in range(num_bins)]
    bin_sums = [0.0] * num_bins
    
    # 分配items
    for prompt_id, weight in items_sorted:
        # 找到放入后总和最小的bin（Best Fit）
        best_bin_idx = min(range(num_bins), 
                          key=lambda i: bin_sums[i] + weight)
        bins[best_bin_idx].append((prompt_id, weight))
        bin_sums[best_bin_idx] += weight
    
    return bins, bin_sums


def schedule_prompts(json_file: str, num_replicas: int = None, output_file: str = None):
    """
    从JSON文件中读取prompt数据，执行bin packing调度
    
    Args:
        json_file: 包含per-prompt avg response lengths的JSON文件路径
        num_replicas: Replicas数量（如果为None，从JSON中读取）
        output_file: 输出文件路径（如果为None，自动生成）
    
    Returns:
        schedule_result: 调度结果字典
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取数据
    if 'all_per_prompt_avg_response_lengths' not in data:
        raise ValueError("JSON文件中未找到all_per_prompt_avg_response_lengths数据")
    if 'all_per_prompt_prompt_lengths' not in data:
        raise ValueError("JSON文件中未找到all_per_prompt_prompt_lengths数据")
    
    avg_lengths = data['all_per_prompt_avg_response_lengths']
    prompt_lengths = data['all_per_prompt_prompt_lengths']
    
    # 获取replicas数量
    if num_replicas is None:
        summary = data.get('summary', {})
        num_replicas = summary.get('num_replicas', 8)
        print(f"从JSON中读取replicas数量: {num_replicas}")
    
    # 准备数据
    items = []
    prompt_stats = {}
    total_prompt_tokens = 0.0
    total_response_tokens = 0.0
    
    for pid_str, avg_resp_len in sorted(avg_lengths.items(), key=lambda x: int(x[0])):
        prompt_len = float(prompt_lengths.get(pid_str, 0.0))
        avg_resp_len = float(avg_resp_len)
        total_len = prompt_len + avg_resp_len
        
        pid = int(pid_str)
        items.append((pid, total_len))
        prompt_stats[pid] = {
            'prompt_length': prompt_len,
            'avg_response_length': avg_resp_len,
            'estimated_total_tokens': total_len,
        }
        total_prompt_tokens += prompt_len
        total_response_tokens += avg_resp_len
    
    total_tokens = total_prompt_tokens + total_response_tokens
    ideal_tokens_per_replica = total_tokens / num_replicas
    
    print("=" * 70)
    print("Bin Packing调度算法")
    print("=" * 70)
    print(f"\n【输入数据】")
    print(f"Prompts数量: {len(items)}")
    print(f"Replicas数量: {num_replicas}")
    print(f"总prompt tokens: {total_prompt_tokens:.0f}")
    print(f"总response tokens: {total_response_tokens:.0f}")
    print(f"估计总tokens(prompt+response): {total_tokens:.0f}")
    print(f"理想平均每个replica: {ideal_tokens_per_replica:.0f} tokens")
    
    # 执行调度
    bins, bin_sums = best_fit_decreasing(items, num_replicas)
    
    print(f"\n【调度结果】")
    for i, (bin_items, bin_sum) in enumerate(zip(bins, bin_sums)):
        deviation = bin_sum - ideal_tokens_per_replica
        deviation_pct = (deviation / ideal_tokens_per_replica) * 100 if ideal_tokens_per_replica > 0 else 0
        print(f"  Replica {i}: {len(bin_items)} prompts, {bin_sum:.0f} tokens "
              f"(偏差: {deviation:+.0f}, {deviation_pct:+.2f}%)")
    
    print(f"\n【负载均衡统计】")
    print(f"  最小: {min(bin_sums):.0f} tokens")
    print(f"  最大: {max(bin_sums):.0f} tokens")
    print(f"  平均: {np.mean(bin_sums):.2f} tokens")
    print(f"  标准差: {np.std(bin_sums):.2f} tokens")
    print(f"  变异系数: {np.std(bin_sums) / np.mean(bin_sums) * 100:.2f}%")
    print(f"  最大偏差: {max(bin_sums) - min(bin_sums):.0f} tokens")
    print(f"  负载不均衡度: {(max(bin_sums) - min(bin_sums)) / np.mean(bin_sums) * 100:.2f}%")
    
    # 构建调度结果
    schedule_result = {
        'algorithm': 'Best Fit Decreasing',
        'num_replicas': num_replicas,
        'num_prompts': len(items),
        'total_tokens': total_tokens,
        'ideal_tokens_per_replica': ideal_tokens_per_replica,
        'total_prompt_tokens': total_prompt_tokens,
        'total_response_tokens': total_response_tokens,
        'replicas': [
            {
                'replica_id': i,
                'prompt_ids': [pid for pid, _ in bin_items],
                'prompts': [
                    {
                        'prompt_id': pid,
                        'prompt_length': prompt_stats[pid]['prompt_length'],
                        'avg_response_length': prompt_stats[pid]['avg_response_length'],
                        'estimated_total_tokens': prompt_stats[pid]['estimated_total_tokens'],
                    }
                    for pid, _ in bin_items
                ],
                'total_tokens': float(bin_sum),
                'num_prompts': len(bin_items),
                'avg_tokens_per_prompt': float(bin_sum / len(bin_items)) if len(bin_items) > 0 else 0,
                'deviation_from_ideal': float(bin_sum - ideal_tokens_per_replica),
                'deviation_percent': float((bin_sum - ideal_tokens_per_replica) / ideal_tokens_per_replica * 100) if ideal_tokens_per_replica > 0 else 0
            }
            for i, (bin_items, bin_sum) in enumerate(zip(bins, bin_sums))
        ],
        'load_balance_stats': {
            'min_tokens': float(min(bin_sums)),
            'max_tokens': float(max(bin_sums)),
            'avg_tokens': float(np.mean(bin_sums)),
            'std_tokens': float(np.std(bin_sums)),
            'cv_percent': float(np.std(bin_sums) / np.mean(bin_sums) * 100),
            'max_deviation': float(max(bin_sums) - min(bin_sums)),
            'imbalance_ratio_percent': float((max(bin_sums) - min(bin_sums)) / np.mean(bin_sums) * 100)
        }
    }
    
    # 保存结果
    if output_file is None:
        output_file = json_file.replace('.json', '_schedule.json')
    
    with open(output_file, 'w') as f:
        json.dump(schedule_result, f, indent=2)
    
    print(f"\n✅ 调度结果已保存到: {output_file}")
    print("=" * 70)
    
    return schedule_result


def main():
    parser = argparse.ArgumentParser(description='Bin Packing调度算法：将prompts分配到replicas')
    parser.add_argument('json_file', type=str, help='包含per-prompt avg response lengths的JSON文件路径')
    parser.add_argument('--num-replicas', type=int, default=None, 
                       help='Replicas数量（如果未指定，从JSON中读取）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（如果未指定，自动生成）')
    
    args = parser.parse_args()
    
    schedule_prompts(args.json_file, args.num_replicas, args.output)


if __name__ == '__main__':
    main()


