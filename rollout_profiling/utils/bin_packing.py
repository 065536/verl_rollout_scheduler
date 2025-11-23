#!/usr/bin/env python3
"""
Bin Packing调度器 - 根据预测的响应长度优化worker负载均衡
"""

import numpy as np
from typing import List, Dict


class BinPackingScheduler:
    """使用First Fit Decreasing算法进行负载均衡"""
    
    def __init__(self, num_workers: int = 16):
        self.num_workers = num_workers
        self.worker_loads = [0.0] * num_workers  # 每个worker的预计总tokens
        self.worker_assignments = [[] for _ in range(num_workers)]  # 每个worker分配的prompt_ids
    
    def schedule_prompts(self, prompt_lengths: Dict[int, float]) -> List[List[int]]:
        """
        使用First Fit Decreasing算法分配prompts
        
        Args:
            prompt_lengths: {prompt_id: avg_length} 字典
            
        Returns:
            assignments: 每个worker分配的prompt_id列表
        """
        # 按长度降序排列
        sorted_prompts = sorted(prompt_lengths.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*70}")
        print(f"🎯 Bin Packing调度")
        print(f"{'='*70}")
        print(f"总Prompts: {len(sorted_prompts)}")
        print(f"Workers: {self.num_workers}")
        if sorted_prompts:
            print(f"最长prompt: {sorted_prompts[0][1]:.0f} tokens (ID: {sorted_prompts[0][0]})")
            print(f"最短prompt: {sorted_prompts[-1][1]:.0f} tokens (ID: {sorted_prompts[-1][0]})")
        
        # 重置
        self.worker_loads = [0.0] * self.num_workers
        self.worker_assignments = [[] for _ in range(self.num_workers)]
        
        # 贪心分配：每次将prompt分配给当前负载最轻的worker
        for prompt_id, length in sorted_prompts:
            # 找到负载最轻的worker
            min_worker = np.argmin(self.worker_loads)
            
            # 分配
            self.worker_assignments[min_worker].append(prompt_id)
            self.worker_loads[min_worker] += length
        
        # 打印统计
        print(f"\n负载分配结果:")
        for i in range(self.num_workers):
            print(f"  Worker {i:2d}: {len(self.worker_assignments[i]):3d} prompts, "
                  f"预计 {self.worker_loads[i]:8.0f} tokens")
        
        print(f"\n负载均衡指标:")
        mean_load = np.mean(self.worker_loads)
        std_load = np.std(self.worker_loads)
        max_load = np.max(self.worker_loads)
        min_load = np.min(self.worker_loads)
        
        print(f"  平均负载: {mean_load:.0f} tokens")
        print(f"  标准差: {std_load:.0f} tokens ({std_load/mean_load*100:.1f}%)")
        print(f"  最大负载: {max_load:.0f} tokens")
        print(f"  最小负载: {min_load:.0f} tokens")
        print(f"  不平衡度: {(max_load - min_load)/mean_load*100:.1f}%")
        if min_load > 0:
            print(f"  最大/最小比: {max_load/min_load:.2f}x")
        
        return self.worker_assignments
    
    def get_statistics(self) -> Dict:
        """获取调度统计信息"""
        mean_load = np.mean(self.worker_loads)
        std_load = np.std(self.worker_loads)
        max_load = np.max(self.worker_loads)
        min_load = np.min(self.worker_loads)
        
        return {
            'mean_load': mean_load,
            'std_load': std_load,
            'max_load': max_load,
            'min_load': min_load,
            'load_imbalance_pct': (max_load - min_load) / mean_load * 100 if mean_load > 0 else 0.0,
            'max_min_ratio': max_load / min_load if min_load > 0 else 0.0
        }

