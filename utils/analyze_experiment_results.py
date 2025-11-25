#!/usr/bin/env python3
"""
分析批量调度实验结果，统计和对比三组实验（default, shuffle, bin_packing）的关键指标
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

def load_metrics_file(filepath: str) -> Optional[Dict]:
    """加载指标JSON文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_summary_stats(metrics: Dict) -> Dict:
    """从指标文件中提取关键统计信息"""
    summary = metrics.get('summary', {})
    
    return {
        'total_rollout_time_s': summary.get('total_rollout_time_s', 0),
        'total_response_tokens': summary.get('total_response_tokens', 0),
        'total_prompt_tokens': summary.get('total_prompt_tokens', 0),
        'num_batches': summary.get('num_batches', 0),
        'num_workers': summary.get('num_workers', 0),
        'num_replicas': summary.get('num_replicas', 0),
        'fastest_worker_time_s': summary.get('fastest_worker_time_s', 0),
        'slowest_worker_time_s': summary.get('slowest_worker_time_s', 0),
        'worker_time_difference_s': summary.get('worker_time_difference_s', 0),
        'throughput_tokens_per_sec': summary.get('total_response_tokens', 0) / (summary.get('total_rollout_time_s', 1) + 1e-9),
        'response_length_mean': summary.get('response_length_stats', {}).get('mean', 0),
        'response_length_std': summary.get('response_length_stats', {}).get('std', 0),
        'response_length_max': summary.get('response_length_stats', {}).get('max', 0),
    }

def analyze_per_worker_stats(metrics: Dict) -> Dict:
    """分析每个worker的统计信息"""
    per_worker_summary = metrics.get('per_worker_summary', [])
    
    if not per_worker_summary:
        return {}
    
    worker_times = [w.get('total_time_s', 0) for w in per_worker_summary]
    worker_tokens = [w.get('total_tokens', 0) for w in per_worker_summary]
    
    return {
        'worker_time_mean': statistics.mean(worker_times) if worker_times else 0,
        'worker_time_std': statistics.stdev(worker_times) if len(worker_times) > 1 else 0,
        'worker_time_min': min(worker_times) if worker_times else 0,
        'worker_time_max': max(worker_times) if worker_times else 0,
        'worker_time_cv': (statistics.stdev(worker_times) / statistics.mean(worker_times)) if worker_times and statistics.mean(worker_times) > 0 else 0,
        'worker_tokens_mean': statistics.mean(worker_tokens) if worker_tokens else 0,
        'worker_tokens_std': statistics.stdev(worker_tokens) if len(worker_tokens) > 1 else 0,
        'worker_tokens_cv': (statistics.stdev(worker_tokens) / statistics.mean(worker_tokens)) if worker_tokens and statistics.mean(worker_tokens) > 0 else 0,
    }

def analyze_per_replica_stats(metrics: Dict) -> Dict:
    """分析每个replica的统计信息"""
    per_replica_summary = metrics.get('per_replica_summary', [])
    
    if not per_replica_summary:
        return {}
    
    replica_times = [r.get('total_wall_clock_time_s', r.get('total_time_s', 0)) for r in per_replica_summary]
    replica_tokens = [r.get('total_tokens', 0) for r in per_replica_summary]
    
    return {
        'replica_time_mean': statistics.mean(replica_times) if replica_times else 0,
        'replica_time_std': statistics.stdev(replica_times) if len(replica_times) > 1 else 0,
        'replica_time_min': min(replica_times) if replica_times else 0,
        'replica_time_max': max(replica_times) if replica_times else 0,
        'replica_time_cv': (statistics.stdev(replica_times) / statistics.mean(replica_times)) if replica_times and statistics.mean(replica_times) > 0 else 0,
        'replica_tokens_mean': statistics.mean(replica_tokens) if replica_tokens else 0,
        'replica_tokens_std': statistics.stdev(replica_tokens) if len(replica_tokens) > 1 else 0,
        'replica_tokens_cv': (statistics.stdev(replica_tokens) / statistics.mean(replica_tokens)) if replica_tokens and statistics.mean(replica_tokens) > 0 else 0,
    }

def analyze_experiment_group(exp_dir: Path, mode: str) -> List[Dict]:
    """分析一个实验组的所有运行"""
    results = []
    
    for run_dir in sorted(exp_dir.glob('run_*')):
        metrics_files = list(run_dir.glob('rollout_worker_metrics_*.json'))
        if not metrics_files:
            print(f"Warning: No metrics file found in {run_dir}")
            continue
        
        metrics_file = metrics_files[0]  # Take the first one
        metrics = load_metrics_file(str(metrics_file))
        
        if not metrics:
            continue
        
        summary_stats = extract_summary_stats(metrics)
        worker_stats = analyze_per_worker_stats(metrics)
        replica_stats = analyze_per_replica_stats(metrics)
        
        run_num = run_dir.name.replace('run_', '')
        results.append({
            'mode': mode,
            'run': run_num,
            'metrics_file': str(metrics_file),
            **summary_stats,
            **worker_stats,
            **replica_stats,
        })
    
    return results

def print_comparison_table(all_results: Dict[str, List[Dict]]):
    """打印对比表格"""
    print("\n" + "="*100)
    print("实验对比统计表")
    print("="*100)
    
    # 按模式分组统计
    for mode in ['default', 'shuffle', 'bin_packing']:
        if mode not in all_results or not all_results[mode]:
            print(f"\n{mode.upper()}: 无数据或实验失败")
            continue
        
        results = all_results[mode]
        print(f"\n{mode.upper()} 模式 ({len(results)} 次运行):")
        print("-" * 100)
        
        # 计算平均值
        avg_total_time = statistics.mean([r['total_rollout_time_s'] for r in results])
        avg_worker_diff = statistics.mean([r['worker_time_difference_s'] for r in results])
        avg_throughput = statistics.mean([r['throughput_tokens_per_sec'] for r in results])
        avg_worker_time_cv = statistics.mean([r.get('worker_time_cv', 0) for r in results])
        avg_replica_time_cv = statistics.mean([r.get('replica_time_cv', 0) for r in results])
        avg_replica_tokens_cv = statistics.mean([r.get('replica_tokens_cv', 0) for r in results])
        
        print(f"  总耗时 (平均): {avg_total_time:.2f}s")
        print(f"  Worker时间差异 (平均): {avg_worker_diff:.2f}s")
        print(f"  吞吐量 (平均): {avg_throughput:.1f} tokens/s")
        print(f"  Worker时间变异系数 (CV, 平均): {avg_worker_time_cv:.3f} (越小越均匀)")
        print(f"  Replica时间变异系数 (CV, 平均): {avg_replica_time_cv:.3f} (越小越均匀)")
        print(f"  Replica tokens变异系数 (CV, 平均): {avg_replica_tokens_cv:.3f} (越小越均匀)")
        
        # 显示每次运行的详细信息
        print(f"\n  各次运行详情:")
        for r in results:
            print(f"    Run {r['run']}: "
                  f"总耗时={r['total_rollout_time_s']:.2f}s, "
                  f"Worker差异={r['worker_time_difference_s']:.2f}s, "
                  f"Replica时间CV={r.get('replica_time_cv', 0):.3f}, "
                  f"Replica tokens CV={r.get('replica_tokens_cv', 0):.3f}")
    
    # 跨模式对比
    print("\n" + "="*100)
    print("跨模式对比 (平均值)")
    print("="*100)
    
    comparison_data = []
    for mode in ['default', 'shuffle', 'bin_packing']:
        if mode not in all_results or not all_results[mode]:
            continue
        
        results = all_results[mode]
        comparison_data.append({
            'mode': mode,
            'avg_total_time': statistics.mean([r['total_rollout_time_s'] for r in results]),
            'avg_worker_diff': statistics.mean([r['worker_time_difference_s'] for r in results]),
            'avg_throughput': statistics.mean([r['throughput_tokens_per_sec'] for r in results]),
            'avg_worker_cv': statistics.mean([r.get('worker_time_cv', 0) for r in results]),
            'avg_replica_time_cv': statistics.mean([r.get('replica_time_cv', 0) for r in results]),
            'avg_replica_tokens_cv': statistics.mean([r.get('replica_tokens_cv', 0) for r in results]),
        })
    
    if comparison_data:
        print(f"{'模式':<15} {'总耗时(s)':<12} {'Worker差异(s)':<15} {'吞吐量(tokens/s)':<18} {'Replica时间CV':<15} {'Replica tokens CV':<18}")
        print("-" * 100)
        for data in comparison_data:
            print(f"{data['mode']:<15} {data['avg_total_time']:<12.2f} {data['avg_worker_diff']:<15.2f} "
                  f"{data['avg_throughput']:<18.1f} {data['avg_replica_time_cv']:<15.3f} {data['avg_replica_tokens_cv']:<18.3f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_experiment_results.py <experiment_dir>")
        print("Example: python analyze_experiment_results.py logs/batch_scheduling_experiments_20251124_150459")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"分析实验目录: {exp_dir}")
    
    all_results = {}
    
    # 分析三组实验
    for mode in ['default', 'shuffle', 'bin_packing']:
        mode_dir = exp_dir / mode
        if not mode_dir.exists():
            print(f"Warning: {mode} directory not found")
            continue
        
        results = analyze_experiment_group(mode_dir, mode)
        if results:
            all_results[mode] = results
            print(f"\n✓ {mode}: 找到 {len(results)} 次运行")
        else:
            print(f"\n✗ {mode}: 未找到有效数据")
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 保存详细结果到JSON
    output_file = exp_dir / 'experiment_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n详细分析结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

