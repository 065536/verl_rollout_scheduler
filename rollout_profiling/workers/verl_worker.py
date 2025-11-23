#!/usr/bin/env python3
"""
VERL框架版本的Rollout Worker
使用VERL的ActorRolloutRefWorker
"""

import os
import time
import ray
from typing import List, Dict

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl"))

from rollout_profiling.utils.verl_utils import (
    VERL_AVAILABLE,
    create_verl_rollout_config,
    setup_verl_environment,
    prompts_to_dataproto,
    dataproto_to_responses,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@ray.remote(num_gpus=1)
class VERLRolloutWorker:
    """使用VERL框架的Rollout Worker"""
    
    def __init__(
        self,
        worker_id: int,
        model_path: str,
        max_tokens: int = 8192,
        gpu_memory: float = 0.5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
    ):
        self.worker_id = worker_id
        self.model_path = model_path
        self.max_tokens = max_tokens
        
        if not VERL_AVAILABLE:
            raise RuntimeError("VERL not available, cannot use VERLRolloutWorker")
        
        # 设置环境变量
        setup_verl_environment()

        # 强制单进程分布式配置，避免继承外部WORLD_SIZE/RANK
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        
        # 为每个worker设置独立的端口（避免冲突）
        base_port = 29500
        os.environ["MASTER_PORT"] = str(base_port + worker_id)
        
        print(f"[VERL Worker {worker_id}] 初始化VERL Rollout Worker...")
        
        # 创建VERL配置
        self.config = create_verl_rollout_config(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            gpu_memory=gpu_memory,
        )
        
        # 初始化VERL worker (role="rollout"表示只做rollout，不做actor训练)
        try:
            self.verl_worker = ActorRolloutRefWorker(config=self.config, role="rollout")
            self.verl_worker.init_model()
            self.tokenizer = self.verl_worker.tokenizer
            print(f"[VERL Worker {worker_id}] ✓ VERL worker初始化完成")
        except Exception as e:
            print(f"[VERL Worker {worker_id}] ❌ VERL worker初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_single(self, prompts, temperature=1.0, top_k=-1, top_p=1.0):
        """为每个prompt生成单个response"""
        return self.generate(prompts, n_samples=1, temperature=temperature, top_k=top_k, top_p=top_p)
    
    def generate(self, prompts, n_samples=16, temperature=1.0, top_k=-1, top_p=1.0):
        """生成responses（使用VERL的generate_sequences）"""
        worker_start_time = time.time()
        
        # 更新配置中的采样参数（如果需要）
        if temperature != self.config.rollout.temperature:
            self.config.rollout.temperature = temperature
        if top_k > 0 and top_k != self.config.rollout.get("top_k", -1):
            self.config.rollout.top_k = top_k
        if top_p < 1.0 and top_p != self.config.rollout.get("top_p", 1.0):
            self.config.rollout.top_p = top_p
        
        # 转换为DataProto
        prompt_ids = list(range(len(prompts)))
        input_dataproto = prompts_to_dataproto(
            prompts=prompts,
            tokenizer=self.tokenizer,
            max_prompt_length=4096,
        )
        
        # 记录原始prompt长度
        original_prompt_lengths = []
        for i in range(len(prompts)):
            if input_dataproto.batch is not None and "attention_mask" in input_dataproto.batch:
                prompt_len = int(input_dataproto.batch["attention_mask"][i].sum().item())
                original_prompt_lengths.append(prompt_len)
            else:
                original_prompt_lengths.append(0)
        
        # 调用VERL的generate_sequences
        pure_generation_start = time.time()
        output_dataproto = self.verl_worker.generate_sequences(input_dataproto)
        pure_generation_end = time.time()
        pure_generation_duration = pure_generation_end - pure_generation_start
        
        worker_end_time = time.time()
        worker_duration = worker_end_time - worker_start_time
        
        # 转换为response格式
        responses = dataproto_to_responses(
            dataproto=output_dataproto,
            tokenizer=self.tokenizer,
            prompt_ids=prompt_ids,
            original_prompt_lengths=original_prompt_lengths,
        )
        
        results = {
            'worker_id': self.worker_id,
            'worker_start_time': worker_start_time,
            'worker_end_time': worker_end_time,
            'worker_duration': worker_duration,
            'pure_generation_duration': pure_generation_duration,
            'overhead_duration': worker_duration - pure_generation_duration,
            'num_prompts': len(prompts),
            'num_responses': len(prompts) * n_samples,
            'responses': responses,
        }
        
        return results
    
    def process_task_queue(
        self,
        scheduler,
        prompts,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        wait_interval=0.2,
    ):
        """从scheduler拉取任务并实时汇报结果"""
        processed = 0
        while True:
            task = ray.get(scheduler.get_next_task.remote(self.worker_id))
            status = task.get('status')
            if status == 'done':
                break
            if status == 'wait':
                time.sleep(wait_interval)
                continue

            prompt_id = task['prompt_id']
            round_num = task['round']
            prompt = prompts[prompt_id]

            result = self.generate_single(
                [prompt],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if result['responses']:
                resp = result['responses'][0]
                response_length = resp['response_length']
                response_text = resp.get('response_text', '')
                prompt_length = resp['prompt_length']
            else:
                response_length = 0
                response_text = ''
                prompt_length = 0

            ray.get(scheduler.report_result.remote(
                worker_id=self.worker_id,
                prompt_id=prompt_id,
                round_num=round_num,
                prompt_length=prompt_length,
                response_length=response_length,
                response_text=response_text,
                worker_duration=result['worker_duration'],
                pure_generation_time=result['pure_generation_duration'],
            ))
            processed += 1

        return {
            'worker_id': self.worker_id,
            'processed_tasks': processed,
        }

