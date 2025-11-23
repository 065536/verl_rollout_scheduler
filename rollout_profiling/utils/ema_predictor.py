#!/usr/bin/env python3
"""
EMA-based response length predictor for load balancing
"""

import numpy as np
import threading
from typing import Dict


class EMALengthPredictor:
    """
    EMA-based response length predictor for load balancing
    
    维护N个prompt的expected response length，使用EMA平滑更新
    """
    
    def __init__(self, num_prompts: int, initial_length: float = 512.0, alpha: float = 0.3):
        """
        Args:
            num_prompts: prompt总数
            initial_length: 初始预测长度
            alpha: EMA平滑系数（0-1），越大对新值越敏感
        """
        self.num_prompts = num_prompts
        self.alpha = alpha
        # 初始化所有prompt的expected length为初始值
        self.expected_lengths = np.full(num_prompts, initial_length, dtype=np.float32)
        # 记录每个prompt已生成的response数量（用于统计）
        self.response_counts = np.zeros(num_prompts, dtype=np.int32)
        # 记录每个prompt的实际平均长度（用于验证）
        self.actual_avg_lengths = np.zeros(num_prompts, dtype=np.float32)
        self.lock = threading.Lock()  # 线程安全
    
    def get_expected_length(self, prompt_id: int) -> float:
        """获取prompt的expected response length"""
        return float(self.expected_lengths[prompt_id])
    
    def get_all_expected_lengths(self) -> Dict[int, float]:
        """获取所有prompt的expected lengths（用于bin packing）"""
        with self.lock:
            return {i: float(self.expected_lengths[i]) for i in range(self.num_prompts)}
    
    def update(self, prompt_id: int, actual_length: int):
        """
        使用EMA更新prompt的expected length
        
        Args:
            prompt_id: prompt ID
            actual_length: 实际生成的response长度
        """
        with self.lock:
            old_expected = self.expected_lengths[prompt_id]
            # EMA更新: new = alpha * actual + (1 - alpha) * old
            new_expected = self.alpha * actual_length + (1 - self.alpha) * old_expected
            self.expected_lengths[prompt_id] = new_expected
            
            # 更新统计信息
            count = self.response_counts[prompt_id]
            if count == 0:
                self.actual_avg_lengths[prompt_id] = actual_length
            else:
                # 移动平均
                self.actual_avg_lengths[prompt_id] = (
                    (self.actual_avg_lengths[prompt_id] * count + actual_length) / (count + 1)
                )
            self.response_counts[prompt_id] += 1
    
    def batch_update(self, updates: Dict[int, int]):
        """批量更新多个prompt的expected length"""
        with self.lock:
            for prompt_id, actual_length in updates.items():
                old_expected = self.expected_lengths[prompt_id]
                new_expected = self.alpha * actual_length + (1 - self.alpha) * old_expected
                self.expected_lengths[prompt_id] = new_expected
                
                count = self.response_counts[prompt_id]
                if count == 0:
                    self.actual_avg_lengths[prompt_id] = actual_length
                else:
                    self.actual_avg_lengths[prompt_id] = (
                        (self.actual_avg_lengths[prompt_id] * count + actual_length) / (count + 1)
                    )
                self.response_counts[prompt_id] += 1
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            valid_counts = self.response_counts > 0
            actual_mean = (
                float(np.mean(self.actual_avg_lengths[valid_counts]))
                if np.any(valid_counts)
                else 0.0
            )
            
            return {
                'expected_mean': float(np.mean(self.expected_lengths)),
                'expected_std': float(np.std(self.expected_lengths)),
                'expected_min': float(np.min(self.expected_lengths)),
                'expected_max': float(np.max(self.expected_lengths)),
                'actual_mean': actual_mean,
                'total_responses': int(np.sum(self.response_counts)),
            }

    def export_state(self) -> Dict[str, list]:
        """导出当前EMA状态用于scheduler"""
        with self.lock:
            return {
                'expected_lengths': self.expected_lengths.copy().tolist(),
                'actual_avg_lengths': self.actual_avg_lengths.copy().tolist(),
                'response_counts': self.response_counts.copy().tolist(),
                'alpha': self.alpha,
            }

