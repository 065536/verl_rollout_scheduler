#!/usr/bin/env python3
"""
生成三个表格，统计default、shuffle、bin_packing三个模式下每个replica的运行时间和response token数
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

def extract_replica_metrics(metrics_file: str) -> Dict[int, Dict]:
    """
    从metrics文件中提取每个replica的运行时间和token数
    
    Returns:
        {replica_id: {'time': float, 'tokens': int, 'num_workers': int}}
    """
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # 方法1: 尝试从per_batch_metrics中的all_replicas获取
    per_batch_metrics = data.get('per_batch_metrics', [])
    replica_data = {}
    
    for batch_metrics in per_batch_metrics:
        # 先尝试all_replicas
        all_replicas = batch_metrics.get('all_replicas', [])
        
        if all_replicas:
            for replica_info in all_replicas:
                replica_id = replica_info.get('replica_id', 0)
                wall_clock_time = replica_info.get('wall_clock_time', replica_info.get('rollout_time', 0))
                response_tokens = replica_info.get('response_tokens', 0)
                num_workers = replica_info.get('num_workers', 2)
                
                if replica_id not in replica_data:
                    replica_data[replica_id] = {
                        'times': [],
                        'tokens': [],
                        'num_workers': num_workers,
                    }
                
                replica_data[replica_id]['times'].append(wall_clock_time)
                replica_data[replica_id]['tokens'].append(response_tokens)
        else:
            # 方法2: 从all_workers中聚合
            all_workers = batch_metrics.get('all_workers', [])
            tp_size = batch_metrics.get('tp_size', 2)
            
            # 按replica_id分组
            replica_workers = {}
            for worker in all_workers:
                replica_id = worker.get('replica_id', worker.get('worker_rank', 0) // tp_size)
                if replica_id not in replica_workers:
                    replica_workers[replica_id] = []
                replica_workers[replica_id].append(worker)
            
            # 计算每个replica的时间和token
            for replica_id, workers in replica_workers.items():
                # 使用最慢的worker时间作为replica时间
                times = [w.get('rollout_time', 0) for w in workers]
                tokens_list = [w.get('response_tokens', 0) for w in workers]
                
                # 同一个replica的workers应该有相同的tokens
                replica_tokens = tokens_list[0] if tokens_list else 0
                replica_time = max(times) if times else 0
                
                if replica_id not in replica_data:
                    replica_data[replica_id] = {
                        'times': [],
                        'tokens': [],
                        'num_workers': len(workers),
                    }
                
                replica_data[replica_id]['times'].append(replica_time)
                replica_data[replica_id]['tokens'].append(replica_tokens)
    
    # 计算平均值（如果有多个batch）
    result = {}
    for replica_id, data in replica_data.items():
        avg_time = statistics.mean(data['times']) if data['times'] else 0
        avg_tokens = statistics.mean(data['tokens']) if data['tokens'] else 0
        
        result[replica_id] = {
            'time_s': avg_time,
            'tokens': int(avg_tokens),
            'num_workers': data['num_workers'],
        }
    
    return result

def generate_table(replica_metrics: Dict[int, Dict], mode: str, run_num: int = None, note: str = None) -> str:
    """
    生成Markdown表格
    
    Args:
        replica_metrics: {replica_id: {'time_s': float, 'tokens': int}}
        mode: 模式名称 (default/shuffle/bin_packing)
        run_num: 运行编号（可选）
    
    Returns:
        Markdown格式的表格字符串
    """
    if not replica_metrics:
        return f"## {mode.upper()} 模式\n\n无数据\n"
    
    # 排序replica_id
    sorted_replicas = sorted(replica_metrics.keys())
    
    # 计算统计信息
    times = [replica_metrics[r]['time_s'] for r in sorted_replicas]
    tokens = [replica_metrics[r]['tokens'] for r in sorted_replicas]
    
    avg_time = statistics.mean(times) if times else 0
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    
    avg_tokens = statistics.mean(tokens) if tokens else 0
    std_tokens = statistics.stdev(tokens) if len(tokens) > 1 else 0
    min_tokens = min(tokens) if tokens else 0
    max_tokens = max(tokens) if tokens else 0
    
    # 生成表格
    title = f"## {mode.upper()} 模式"
    if run_num is not None:
        title += f" (Run {run_num})"
    title += "\n\n"
    
    table = title
    table += "| Replica ID | 运行时间 (s) | Response Tokens | 吞吐量 (tokens/s) |\n"
    table += "|------------|--------------|-----------------|---------------------|\n"
    
    for replica_id in sorted_replicas:
        metrics = replica_metrics[replica_id]
        time_s = metrics['time_s']
        tokens = metrics['tokens']
        throughput = tokens / (time_s + 1e-9)
        
        table += f"| {replica_id} | {time_s:.2f} | {tokens:,} | {throughput:.1f} |\n"
    
    # 添加统计信息
    table += "\n**统计信息:**\n"
    table += f"- 运行时间: 平均={avg_time:.2f}s, 标准差={std_time:.2f}s, 最小={min_time:.2f}s, 最大={max_time:.2f}s\n"
    table += f"- Response Tokens: 平均={avg_tokens:,.0f}, 标准差={std_tokens:,.0f}, 最小={min_tokens:,}, 最大={max_tokens:,}\n"
    table += f"- 时间差异: {max_time - min_time:.2f}s\n"
    table += f"- Token差异: {max_tokens - min_tokens:,}\n"
    
    return table

def process_experiment_dir(exp_dir: Path, mode: str) -> List[Tuple[int, Dict]]:
    """
    处理一个模式的所有运行，返回每个运行的replica metrics
    
    Returns:
        [(run_num, replica_metrics), ...]
    """
    mode_dir = exp_dir / mode
    if not mode_dir.exists():
        return []
    
    results = []
    for run_dir in sorted(mode_dir.glob('run_*')):
        metrics_files = list(run_dir.glob('rollout_worker_metrics_*.json'))
        if not metrics_files:
            continue
        
        metrics_file = metrics_files[0]
        run_num = int(run_dir.name.replace('run_', ''))
        
        replica_metrics = extract_replica_metrics(str(metrics_file))
        if replica_metrics:
            results.append((run_num, replica_metrics))
    
    return results

def generate_aggregated_table(all_runs: List[Tuple[int, Dict]], mode: str) -> str:
    """
    生成聚合表格（所有运行的平均值）
    """
    if not all_runs:
        return f"## {mode.upper()} 模式\n\n无数据\n"
    
    # 聚合所有运行的replica数据
    replica_aggregated = {}
    
    for run_num, replica_metrics in all_runs:
        for replica_id, metrics in replica_metrics.items():
            if replica_id not in replica_aggregated:
                replica_aggregated[replica_id] = {
                    'times': [],
                    'tokens': [],
                }
            replica_aggregated[replica_id]['times'].append(metrics['time_s'])
            replica_aggregated[replica_id]['tokens'].append(metrics['tokens'])
    
    # 计算平均值
    result = {}
    for replica_id, data in replica_aggregated.items():
        result[replica_id] = {
            'time_s': statistics.mean(data['times']),
            'tokens': statistics.mean(data['tokens']),
            'time_std': statistics.stdev(data['times']) if len(data['times']) > 1 else 0,
            'tokens_std': statistics.stdev(data['tokens']) if len(data['tokens']) > 1 else 0,
        }
    
    # 生成表格
    sorted_replicas = sorted(result.keys())
    
    title = f"## {mode.upper()} 模式 (所有运行的平均值)\n\n"
    table = title
    table += "| Replica ID | 运行时间 (s) | Response Tokens | 吞吐量 (tokens/s) |\n"
    table += "|------------|--------------|-----------------|---------------------|\n"
    
    for replica_id in sorted_replicas:
        metrics = result[replica_id]
        time_s = metrics['time_s']
        time_std = metrics['time_std']
        tokens = metrics['tokens']
        tokens_std = metrics['tokens_std']
        throughput = tokens / (time_s + 1e-9)
        
        table += f"| {replica_id} | {time_s:.2f} ± {time_std:.2f} | {tokens:,.0f} ± {tokens_std:,.0f} | {throughput:.1f} |\n"
    
    # 计算总体统计
    times = [result[r]['time_s'] for r in sorted_replicas]
    tokens = [result[r]['tokens'] for r in sorted_replicas]
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    avg_tokens = statistics.mean(tokens)
    std_tokens = statistics.stdev(tokens) if len(tokens) > 1 else 0
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    
    table += "\n**统计信息:**\n"
    table += f"- 运行时间: 平均={avg_time:.2f}s, 标准差={std_time:.2f}s, 最小={min_time:.2f}s, 最大={max_time:.2f}s\n"
    table += f"- Response Tokens: 平均={avg_tokens:,.0f}, 标准差={std_tokens:,.0f}, 最小={min_tokens:,}, 最大={max_tokens:,}\n"
    table += f"- 时间差异: {max_time - min_time:.2f}s\n"
    table += f"- Token差异: {max_tokens - min_tokens:,}\n"
    
    return table

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_replica_tables.py <experiment_dir>")
        print("Example: python generate_replica_tables.py logs/batch_scheduling_experiments_20251124_150459")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    output_file = exp_dir / 'replica_metrics_tables.md'
    
    print(f"处理实验目录: {exp_dir}")
    print(f"输出文件: {output_file}")
    
    tables = []
    tables.append("# Replica运行时间和Token统计表\n\n")
    tables.append(f"实验目录: `{exp_dir}`\n\n")
    
    # 处理三个模式
    for mode in ['default', 'shuffle', 'bin_packing']:
        print(f"\n处理 {mode} 模式...")
        all_runs = process_experiment_dir(exp_dir, mode)
        
        if not all_runs:
            tables.append(f"## {mode.upper()} 模式\n\n无数据\n\n")
            continue
        
        # 生成每个运行的表格
        for run_num, replica_metrics in all_runs:
            note = None
            table = generate_table(replica_metrics, mode, run_num, note)
            tables.append(table)
            tables.append("\n")
        
        # 检查所有运行的token数是否完全相同
        if len(all_runs) > 1:
            first_run_tokens = [all_runs[0][1][r]['tokens'] for r in sorted(all_runs[0][1].keys())]
            all_same = True
            for run_num, replica_metrics in all_runs[1:]:
                current_tokens = [replica_metrics[r]['tokens'] for r in sorted(replica_metrics.keys())]
                if current_tokens != first_run_tokens:
                    all_same = False
                    break
            
            if all_same and mode == 'bin_packing':
                aggregated_table = aggregated_table.rstrip() + "\n\n**⚠️ 重要提示：** 所有三个运行的token数完全相同，这是因为vLLM使用了固定的随机种子(seed=0)。即使temperature=0.6，如果随机种子相同，生成的响应也会完全相同。如果需要不同的响应，应该为每次运行设置不同的seed。\n"
        
        # 生成聚合表格
        aggregated_table = generate_aggregated_table(all_runs, mode)
        tables.append(aggregated_table)
        
        # 检查所有运行的token数是否完全相同
        if len(all_runs) > 1:
            first_run_tokens = [all_runs[0][1][r]['tokens'] for r in sorted(all_runs[0][1].keys())]
            all_same = True
            for run_num, replica_metrics in all_runs[1:]:
                current_tokens = [replica_metrics[r]['tokens'] for r in sorted(replica_metrics.keys())]
                if current_tokens != first_run_tokens:
                    all_same = False
                    break
            
            if all_same and mode == 'bin_packing':
                tables.append("\n**⚠️ 重要提示：** 所有三个运行的token数完全相同，这是因为vLLM使用了固定的随机种子(seed=0)。即使temperature=0.6，如果随机种子相同，生成的响应也会完全相同。如果需要不同的响应，应该为每次运行设置不同的seed。\n")
        
        tables.append("\n")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(tables))
    
    print(f"\n✓ 表格已生成: {output_file}")
    
    # 同时在终端显示
    print("\n" + "="*80)
    print(''.join(tables))

if __name__ == '__main__':
    main()

