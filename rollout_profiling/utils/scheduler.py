#!/usr/bin/env python3
"""
任务调度器 - 实时EMA更新 + 任务队列
"""

import ray
from collections import defaultdict
from typing import Dict


@ray.remote(num_cpus=0)
class TaskScheduler:
    """任务队列 + 实时EMA更新"""

    def __init__(
        self,
        ema_state: Dict[str, list],
        remaining_rounds: int,
    ):
        self.alpha = ema_state.get('alpha', 0.3)
        self.expected_lengths = list(ema_state['expected_lengths'])
        self.actual_avg_lengths = list(ema_state['actual_avg_lengths'])
        self.response_counts = list(ema_state['response_counts'])

        self.num_prompts = len(self.expected_lengths)
        self.remaining_rounds = remaining_rounds
        self.remaining = [remaining_rounds] * self.num_prompts
        self.next_round = [1] * self.num_prompts  # 下一次生成的round编号（从1开始）
        self.inflight = [False] * self.num_prompts
        self.pending_total = self.num_prompts * remaining_rounds

        self.pipeline_results = []
        self.worker_stats = defaultdict(lambda: {
            'num_responses': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'pure_generation_time': 0.0,
        })

    def get_next_task(self, worker_id: int) -> Dict:
        if self.pending_total == 0:
            return {'status': 'done'}

        available = [
            pid
            for pid in range(self.num_prompts)
            if self.remaining[pid] > 0 and not self.inflight[pid]
        ]

        if not available:
            return {'status': 'wait'}

        # 选择当前预测长度最大的prompt
        prompt_id = max(available, key=lambda pid: self.expected_lengths[pid])
        self.inflight[prompt_id] = True

        return {
            'status': 'task',
            'prompt_id': prompt_id,
            'round': self.next_round[prompt_id],
        }

    def report_result(
        self,
        worker_id: int,
        prompt_id: int,
        round_num: int,
        prompt_length: int,
        response_length: int,
        response_text: str,
        worker_duration: float,
        pure_generation_time: float,
    ):
        # 更新记录
        self.pipeline_results.append({
            'prompt_id': prompt_id,
            'round': round_num,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'response_text': response_text,
            'worker_id': worker_id,
            'worker_duration': worker_duration,
            'pure_generation_time': pure_generation_time,
        })

        # 更新EMA
        old_expected = self.expected_lengths[prompt_id]
        new_expected = self.alpha * response_length + (1 - self.alpha) * old_expected
        self.expected_lengths[prompt_id] = new_expected

        count = self.response_counts[prompt_id]
        new_actual = (
            (self.actual_avg_lengths[prompt_id] * count + response_length) / (count + 1)
            if count > 0
            else response_length
        )
        self.actual_avg_lengths[prompt_id] = new_actual
        self.response_counts[prompt_id] = count + 1

        # 更新状态
        self.inflight[prompt_id] = False
        self.remaining[prompt_id] -= 1
        self.pending_total -= 1
        self.next_round[prompt_id] += 1

        # 更新worker统计
        stats = self.worker_stats[worker_id]
        stats['num_responses'] += 1
        stats['total_tokens'] += response_length
        stats['total_time'] += worker_duration
        stats['pure_generation_time'] += pure_generation_time

    def get_all_results(self):
        return {
            'responses': self.pipeline_results,
            'ema_state': {
                'expected_lengths': self.expected_lengths,
                'actual_avg_lengths': self.actual_avg_lengths,
                'response_counts': self.response_counts,
                'alpha': self.alpha,
            },
            'worker_stats': dict(self.worker_stats),
        }

