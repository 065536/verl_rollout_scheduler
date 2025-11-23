#!/usr/bin/env python3
"""
VERL框架工具函数
用于VERL版本的worker
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl"))

import torch
from typing import List, Dict, Optional
from omegaconf import OmegaConf, DictConfig

# VERL imports
try:
    from verl import DataProto
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils.model import compute_position_id_with_mask
    VERL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ VERL imports failed: {e}")
    VERL_AVAILABLE = False


def create_verl_rollout_config(
    model_path: str,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    gpu_memory: float = 0.5,
) -> DictConfig:
    """
    创建VERL rollout worker的完整配置
    
    Args:
        model_path: 模型路径
        max_tokens: 最大生成token数
        temperature: 采样温度
        top_k: top-k采样
        top_p: nucleus采样
        gpu_memory: GPU内存使用率（默认0.5，因为actor模型会占用内存）
        
    Returns:
        OmegaConf DictConfig for VERL ActorRolloutRefWorker
    """
    if not VERL_AVAILABLE:
        raise RuntimeError("VERL not available")
    
    # 检测模型大小
    model_name_lower = model_path.lower()
    is_32b_model = "32b" in model_name_lower or "32-b" in model_name_lower
    
    if is_32b_model:
        optimized_gpu_memory = min(gpu_memory * 1.2, 0.95)
        optimized_max_len = 16384
    else:
        optimized_gpu_memory = gpu_memory
        optimized_max_len = min(32768, max_tokens)
    
    config_dict = {
        "model": {
            "path": model_path,
            "trust_remote_code": True,
        },
        "rollout": {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "sync",
            "skip_tokenizer_init": True,
            "response_length": max_tokens,
            "prompt_length": 2048,
            "temperature": temperature,
            "top_k": -1 if top_k <= 0 else top_k,
            "top_p": 1.0 if top_p >= 1.0 else top_p,
            "do_sample": True,
            "n": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": optimized_gpu_memory,
            "ignore_eos": False,
            "enforce_eager": False,
            "free_cache_engine": True,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "max_num_batched_tokens": 32768,
            "max_model_len": optimized_max_len,
            "max_num_seqs": 1024,
            "enable_chunked_prefill": False,
            "load_format": "dummy",
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": None,
            "log_prob_use_dynamic_bsz": False,
            "log_prob_max_token_len_per_gpu": 16384,
            "disable_log_stats": True,
            "multi_stage_wake_up": False,
            "engine_kwargs": {
                "vllm": {},
            },
            "calculate_log_probs": False,
            "profiler": {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "save_path": "/tmp/verl_profiling",
                "tool_config": None,
            },
        },
        "actor": {
            "strategy": "fsdp",
            "fsdp_config": {
                "fsdp_size": 1,
            },
        },
        "nccl_timeout": 600,
    }
    
    return OmegaConf.create(config_dict)


def setup_verl_environment():
    """设置VERL所需的环境变量"""
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["TRUST_REMOTE_CODE"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    
    # 分布式环境变量（单GPU worker时）
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"


def prompts_to_dataproto(
    prompts: List[List[Dict]],
    tokenizer,
    max_prompt_length: int = 4096,
) -> DataProto:
    """
    将prompt列表转换为VERL的DataProto格式
    
    Args:
        prompts: prompt列表（消息格式，list of list of dicts）
        tokenizer: tokenizer对象
        max_prompt_length: 最大prompt长度
        
    Returns:
        DataProto对象
    """
    if not VERL_AVAILABLE:
        raise RuntimeError("VERL not available")
    
    # 格式化prompts
    formatted_prompts = []
    for messages in prompts:
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        except Exception as e:
            print(f"⚠️ Chat template failed: {e}")
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
                formatted_prompts.append(messages[0].get('content', str(messages)))
            else:
                formatted_prompts.append(str(messages))
    
    # Tokenize
    tokenized = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
    )
    
    # 创建DataProto
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)
    
    dataproto = DataProto.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        meta_info={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )
    
    return dataproto


def dataproto_to_responses(
    dataproto: DataProto,
    tokenizer,
    prompt_ids: List[int],
    original_prompt_lengths: Optional[List[int]] = None,
) -> List[Dict]:
    """
    将VERL的DataProto结果转换为response数据格式
    
    Args:
        dataproto: VERL generate_sequences返回的DataProto
        tokenizer: tokenizer对象
        prompt_ids: 对应的prompt ID列表
        original_prompt_lengths: 原始prompt长度列表（如果已知）
        
    Returns:
        response数据列表
    """
    if not VERL_AVAILABLE:
        raise RuntimeError("VERL not available")
    
    responses = []
    batch = dataproto.batch
    
    responses_tokens = batch.get("responses", None)
    input_ids = batch.get("input_ids", None)
    attention_mask = batch.get("attention_mask", None)
    
    if responses_tokens is not None:
        # 直接使用responses字段
        for i, prompt_id in enumerate(prompt_ids):
            response_tokens = responses_tokens[i]
            
            # 移除padding tokens
            if isinstance(response_tokens, torch.Tensor):
                if tokenizer.pad_token_id is not None:
                    valid_mask = response_tokens != tokenizer.pad_token_id
                    valid_tokens = response_tokens[valid_mask]
                else:
                    valid_tokens = response_tokens
                
                response_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                response_length = len(valid_tokens) if len(valid_tokens.shape) == 0 else valid_tokens.shape[0]
            else:
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                response_length = len(response_tokens) if hasattr(response_tokens, '__len__') else 0
            
            # 计算prompt长度
            if original_prompt_lengths and i < len(original_prompt_lengths):
                prompt_length = original_prompt_lengths[i]
            elif attention_mask is not None:
                total_len = int(attention_mask[i].sum().item())
                prompt_length = total_len - response_length
            elif input_ids is not None:
                prompt_length = input_ids.shape[1] - response_length if len(input_ids.shape) > 1 else 0
            else:
                prompt_length = 0
            
            responses.append({
                'prompt_idx': i,
                'sample_idx': 0,
                'prompt_length': prompt_length,
                'response_length': response_length,
                'response_text': response_text,
            })
    elif input_ids is not None and attention_mask is not None:
        # 如果没有responses字段，从input_ids中提取
        for i, prompt_id in enumerate(prompt_ids):
            valid_len = int(attention_mask[i].sum().item())
            
            # 使用原始prompt长度（如果已知）
            if original_prompt_lengths and i < len(original_prompt_lengths):
                prompt_length = original_prompt_lengths[i]
            else:
                # 简化：假设prompt和response各占一半
                prompt_length = valid_len // 2
            
            response_tokens = input_ids[i, prompt_length:valid_len]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            response_length = len(response_tokens)
            
            responses.append({
                'prompt_idx': i,
                'sample_idx': 0,
                'prompt_length': prompt_length,
                'response_length': response_length,
                'response_text': response_text,
            })
    else:
        raise ValueError("DataProto must contain either 'responses' or 'input_ids' field")
    
    return responses

