#!/usr/bin/env python3
"""
详细分析每个replica的token分配和时间分布，解释为什么bin packing导致负载不均衡
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_replica_distribution(metrics_file: str):
    """分析replica级别的token和时间分布"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    per_worker_summary = metrics.get('per_worker_summary', [])
    
    # 按replica分组
    replica_data = defaultdict(lambda: {'workers': [], 'times': [], 'tokens': []})
    
    for worker in per_worker_summary:
        replica_id = worker.get('replica_id', 0)
        replica_data[replica_id]['workers'].append(worker)
        replica_data[replica_id]['times'].append(worker.get('total_time_s', 0))
        replica_data[replica_id]['tokens'].append(worker.get('total_tokens', 0))
    
    # 计算每个replica的统计信息
    replica_stats = []
    for replica_id in sorted(replica_data.keys()):
        data = replica_data[replica_id]
        times = data['times']
        tokens = data['tokens']
        
        # 同一个replica内的worker应该有相同的tokens（TP组内共享）
        replica_tokens = tokens[0] if tokens else 0
        replica_time = max(times) if times else 0  # 取最慢的worker时间
        
        replica_stats.append({
            'replica_id': replica_id,
            'num_workers': len(data['workers']),
            'tokens': replica_tokens,
            'time_s': replica_time,
            'worker_times': times,
            'worker_time_std': statistics.stdev(times) if len(times) > 1 else 0,
        })
    
    return replica_stats

def print_replica_analysis(metrics_file: str, mode: str):
    """打印replica分析结果"""
    print(f"\n{'='*80}")
    print(f"{mode.upper()} 模式 - Replica分布分析")
    print(f"{'='*80}")
    print(f"指标文件: {Path(metrics_file).name}")
    
    stats = analyze_replica_distribution(metrics_file)
    
    if not stats:
        print("无数据")
        return
    
    # 计算总体统计
    replica_times = [s['time_s'] for s in stats]
    replica_tokens = [s['tokens'] for s in stats]
    
    print(f"\nReplica级别统计:")
    print(f"  Replica数量: {len(stats)}")
    print(f"  Token分布:")
    print(f"    平均: {statistics.mean(replica_tokens):.0f}")
    print(f"    标准差: {statistics.stdev(replica_tokens) if len(replica_tokens) > 1 else 0:.0f}")
    print(f"    最小: {min(replica_tokens):.0f}")
    print(f"    最大: {max(replica_tokens):.0f}")
    print(f"    变异系数(CV): {(statistics.stdev(replica_tokens) / statistics.mean(replica_tokens)) if len(replica_tokens) > 1 and statistics.mean(replica_tokens) > 0 else 0:.3f}")
    
    print(f"\n  时间分布:")
    print(f"    平均: {statistics.mean(replica_times):.2f}s")
    print(f"    标准差: {statistics.stdev(replica_times) if len(replica_times) > 1 else 0:.2f}s")
    print(f"    最小: {min(replica_times):.2f}s")
    print(f"    最大: {max(replica_times):.2f}s")
    print(f"    时间差异: {max(replica_times) - min(replica_times):.2f}s")
    print(f"    变异系数(CV): {(statistics.stdev(replica_times) / statistics.mean(replica_times)) if len(replica_times) > 1 and statistics.mean(replica_times) > 0 else 0:.3f}")
    
    # 计算吞吐量
    throughputs = [tokens / (time + 1e-9) for tokens, time in zip(replica_tokens, replica_times)]
    print(f"\n  吞吐量分布 (tokens/s):")
    print(f"    平均: {statistics.mean(throughputs):.1f}")
    print(f"    标准差: {statistics.stdev(throughputs) if len(throughputs) > 1 else 0:.1f}")
    print(f"    最小: {min(throughputs):.1f}")
    print(f"    最大: {max(throughputs):.1f}")
    
    print(f"\n各Replica详情:")
    print(f"{'Replica':<10} {'Tokens':<12} {'Time(s)':<12} {'Throughput':<15} {'Worker时间差异(s)':<20}")
    print("-" * 80)
    for s in sorted(stats, key=lambda x: x['replica_id']):
        throughput = s['tokens'] / (s['time_s'] + 1e-9)
        print(f"{s['replica_id']:<10} {s['tokens']:<12.0f} {s['time_s']:<12.2f} {throughput:<15.1f} {s['worker_time_std']:<20.3f}")
    
    # 分析负载不均衡的原因
    print(f"\n负载均衡分析:")
    token_cv = (statistics.stdev(replica_tokens) / statistics.mean(replica_tokens)) if len(replica_tokens) > 1 and statistics.mean(replica_tokens) > 0 else 0
    time_cv = (statistics.stdev(replica_times) / statistics.mean(replica_times)) if len(replica_times) > 1 and statistics.mean(replica_times) > 0 else 0
    
    if token_cv < 0.01:
        print(f"  ✓ Token分配非常均匀 (CV={token_cv:.3f})")
    elif token_cv < 0.1:
        print(f"  ⚠ Token分配基本均匀 (CV={token_cv:.3f})")
    else:
        print(f"  ✗ Token分配不均匀 (CV={token_cv:.3f})")
    
    if time_cv < 0.05:
        print(f"  ✓ 执行时间非常均匀 (CV={time_cv:.3f})")
    elif time_cv < 0.15:
        print(f"  ⚠ 执行时间基本均匀 (CV={time_cv:.3f})")
    else:
        print(f"  ✗ 执行时间不均匀 (CV={time_cv:.3f})")
        print(f"     原因分析:")
        print(f"     - 虽然token分配可能均匀，但实际响应长度有方差")
        print(f"     - 某些replica分配到的prompt实际响应更长，导致处理时间增加")
        print(f"     - Bin packing基于历史平均，无法预测实际响应长度")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_replica_distribution.py <experiment_dir> [mode]")
        print("Example: python analyze_replica_distribution.py logs/batch_scheduling_experiments_20251124_150459")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    mode_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    for mode in ['default', 'shuffle', 'bin_packing']:
        if mode_filter and mode != mode_filter:
            continue
        
        mode_dir = exp_dir / mode
        if not mode_dir.exists():
            continue
        
        # 取第一个run的结果
        run_dirs = sorted(mode_dir.glob('run_*'))
        if not run_dirs:
            continue
        
        metrics_files = list(run_dirs[0].glob('rollout_worker_metrics_*.json'))
        if not metrics_files:
            continue
        
        print_replica_analysis(str(metrics_files[0]), mode)

if __name__ == '__main__':
    main()

