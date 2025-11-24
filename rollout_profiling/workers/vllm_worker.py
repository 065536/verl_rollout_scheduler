#!/usr/bin/env python3
"""
vLLM直接版本的Rollout Worker
不使用VERL框架，直接调用vLLM（VERL模拟器）
"""

import os
import time
import ray
from typing import List, Dict


@ray.remote(num_gpus=1)
class VLLMRolloutWorker:
    """自定义Rollout Worker（不使用VERL框架，直接使用vLLM）"""
    
    def __init__(self, worker_id: int, model_path: str, max_tokens: int = 8192, gpu_memory: float = 0.8):
        self.worker_id = worker_id
        self.model_path = model_path
        self.max_tokens = max_tokens
        
        # 设置环境变量以避免子进程中的GPU检查问题
        # 这些设置可以帮助vLLM在Ray actor的子进程中正常工作
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # 设置trust_remote_code环境变量，确保vLLM在子进程中也能识别
        os.environ["TRUST_REMOTE_CODE"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        
        # 打印关键的GPU可见性环境变量，便于定位问题
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>")
        ray_acc_ids = os.environ.get("RAY_ACCELERATOR_IDS", "<unset>")
        print(f"[vLLM Worker {worker_id}] CUDA_VISIBLE_DEVICES={cuda_visible}, "
              f"NVIDIA_VISIBLE_DEVICES={nvidia_visible}, "
              f"RAY_ACCELERATOR_IDS={ray_acc_ids}")
        
        # 某些环境下pynvml会读取真正的物理GPU索引，但Ray对子进程做了限流，导致pynvml抛出InvalidArgument。
        # 通过强制vLLM使用Non-NVML平台（直接使用torch.cuda查询）可以规避这一问题。
        force_non_nvml = os.environ.get("VLLM_FORCE_NON_NVML", "1").lower() in ("1", "true", "yes")
        if force_non_nvml:
            try:
                import vllm.platforms.cuda as vllm_cuda
                from vllm import platforms as vllm_platforms
                
                # 只有在当前平台尚未初始化时才能修改
                vllm_platforms._current_platform = None  # type: ignore[attr-defined]
                vllm_cuda.CudaPlatform = vllm_cuda.NonNvmlCudaPlatform
                print(f"[vLLM Worker {worker_id}] ✅ Forcing vLLM Non-NVML CUDA platform")
            except Exception as e:
                print(f"[vLLM Worker {worker_id}] ⚠️ Failed to enforce Non-NVML platform: {e}")
        
        # 初始化vLLM
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoConfig
        
        # 对于某些模型（如DeepSeek-R1-Distill-Qwen-14B），可能需要trust_remote_code=True
        # 先加载config和tokenizer，确保模型架构被正确识别
        print(f"[vLLM Worker {worker_id}] 预加载模型配置...")
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            print(f"[vLLM Worker {worker_id}] ✓ 模型配置加载成功: {config.model_type}")
        except Exception as e:
            print(f"[vLLM Worker {worker_id}] ⚠️  配置加载警告: {e}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 尝试初始化vLLM，如果失败则重试
        # 检测模型大小，32B模型需要特殊配置
        model_name_lower = model_path.lower()
        is_32b_model = "32b" in model_name_lower or "32-b" in model_name_lower
        
        if is_32b_model:
            # 32B模型单卡80GB放不下，需要优化配置
            # 降低max_model_len以减少KV cache需求，提高gpu_memory_utilization
            print(f"[vLLM Worker {worker_id}] 检测到32B模型，使用优化配置")
            optimized_gpu_memory = min(gpu_memory * 1.2, 0.95)  # 提高内存使用率
            optimized_max_len = 16384  # 降低max_model_len以减少KV cache（prompt~4k + response~12k）
        else:
            optimized_gpu_memory = gpu_memory
            optimized_max_len = min(32768, self.max_tokens)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.llm = LLM(
                    model=model_path,
                    trust_remote_code=True,  # 改为True以支持自定义模型架构
                    gpu_memory_utilization=optimized_gpu_memory,
                    tensor_parallel_size=1,
                    max_model_len=optimized_max_len,
                    disable_custom_all_reduce=True,
                    enforce_eager=False,  # 禁用torch compile
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[vLLM Worker {worker_id}] 初始化失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    raise
        
        print(f"[vLLM Worker {worker_id}] 初始化完成")
    
    def generate_single(self, prompts, temperature=1.0, top_k=-1, top_p=1.0):
        """
        为每个prompt生成单个response（用于第一轮和后续轮次）
        
        Args:
            prompts: prompt列表（消息格式，list of list of dicts）
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            
        Returns:
            results: 包含timing和response信息的字典
        """
        return self.generate(prompts, n_samples=1, temperature=temperature, top_k=top_k, top_p=top_p)
    
    def generate(self, prompts, n_samples=16, temperature=1.0, top_k=-1, top_p=1.0):
        """
        生成responses
        
        Args:
            prompts: prompt列表（消息格式，list of list of dicts）
            n_samples: 每个prompt生成n个response
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            
        Returns:
            results: 包含timing和response信息的字典
        """
        from vllm import SamplingParams
        
        worker_start_time = time.time()
        
        # 使用chat template格式化prompts
        formatted_prompts = []
        for idx, messages in enumerate(prompts):
            try:
                # 应用chat template
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
                
                # 只在第一个prompt时输出调试信息
                if idx == 0:
                    print(f"[vLLM Worker {self.worker_id}] ✓ Chat template applied successfully")
                    print(f"[vLLM Worker {self.worker_id}] Sample formatted prompt (first 200 chars):")
                    print(f"[vLLM Worker {self.worker_id}] {formatted[:200]}...")
                    
            except Exception as e:
                # 如果格式化失败，fallback到原始格式
                print(f"[vLLM Worker {self.worker_id}] ⚠️ Chat template failed for prompt {idx}: {e}")
                print(f"[vLLM Worker {self.worker_id}] Prompt type: {type(messages)}, content: {str(messages)[:100]}")
                if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                    formatted_prompts.append(messages[0].get('content', str(messages)))
                else:
                    formatted_prompts.append(str(messages))
        
        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=self.max_tokens,
        )
        
        # 记录纯LLM生成时间（排除格式化和后处理）
        pure_generation_start = time.time()
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        pure_generation_end = time.time()
        pure_generation_duration = pure_generation_end - pure_generation_start
        
        worker_end_time = time.time()
        worker_duration = worker_end_time - worker_start_time
        
        # 处理结果
        results = {
            'worker_id': self.worker_id,
            'worker_start_time': worker_start_time,
            'worker_end_time': worker_end_time,
            'worker_duration': worker_duration,
            'pure_generation_duration': pure_generation_duration,  # 纯LLM推理时间
            'overhead_duration': worker_duration - pure_generation_duration,  # 开销时间（格式化+后处理）
            'num_prompts': len(prompts),
            'num_responses': len(prompts) * n_samples,
            'responses': []
        }
        
        for prompt_idx, (formatted_prompt, output) in enumerate(zip(formatted_prompts, outputs)):
            # Prompt长度（使用格式化后的prompt）
            prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            prompt_length = len(prompt_tokens)
            
            # 每个prompt的n个responses
            for sample_idx, output_sample in enumerate(output.outputs):
                response_text = output_sample.text
                response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
                response_length = len(response_tokens)
                
                results['responses'].append({
                    'prompt_idx': prompt_idx,
                    'sample_idx': sample_idx,
                    'prompt_length': prompt_length,
                    'response_length': response_length,
                    'response_text': response_text,
                })
        
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


