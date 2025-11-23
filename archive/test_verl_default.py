#!/usr/bin/env python3
"""
测试脚本：验证VERL默认调度能否正常运行
使用小规模配置进行快速测试
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl"))

# 设置环境变量
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from rollout_profiling.workers.verl_worker import VERLRolloutWorker
from rollout_profiling.utils.verl_utils import prompts_to_dataproto, dataproto_to_responses


def test_verl_default():
    """测试VERL默认调度"""
    print("=" * 70)
    print("测试 VERL 默认调度")
    print("=" * 70)
    
    # 测试配置
    model_path = os.environ.get("MODEL_PATH", "/data/250010176/codes/models/Qwen3-4B")
    data_file = os.environ.get("DATA_FILE", str(project_root / "data" / "dapo_math_subset_128.parquet"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))  # 小规模测试用2个worker
    max_tokens = int(os.environ.get("MAX_TOKENS", "512"))  # 小token数快速测试
    test_prompts_count = 4  # 只测试4个prompts
    
    print(f"\n测试配置:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_file}")
    print(f"  Workers: {num_workers}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Test prompts: {test_prompts_count}")
    print()
    
    # 初始化Ray
    if not ray.is_initialized():
        print("初始化Ray...")
        ray.init(
            num_cpus=num_workers + 2,
            ignore_reinit_error=True,
        )
        print("✓ Ray初始化完成")
    else:
        print("✓ Ray已初始化")
    
    # 加载数据集（只取前几个prompts）
    print(f"\n加载数据集: {data_file}")
    try:
        df = pd.read_parquet(data_file)
        print(f"✓ 数据集大小: {len(df)} 条")
        
        # 提取prompts
        if "prompt" in df.columns:
            prompts_raw = df["prompt"].tolist()
        elif "messages" in df.columns:
            prompts_raw = df["messages"].tolist()
        else:
            raise ValueError(f"未找到prompt列，可用列: {df.columns.tolist()}")
        
        def normalize_prompt(prompt):
            while True:
                if isinstance(prompt, list):
                    return prompt
                if isinstance(prompt, dict):
                    return [prompt]
                if isinstance(prompt, str):
                    return [{"role": "user", "content": prompt}]
                if isinstance(prompt, np.ndarray):
                    if prompt.size == 0:
                        return None
                    if prompt.size == 1:
                        prompt = prompt.item()
                        continue
                    prompt = prompt.tolist()
                    continue
                if isinstance(prompt, pd.Series):
                    if len(prompt) == 0:
                        return None
                    if len(prompt) == 1:
                        prompt = prompt.iloc[0]
                        continue
                    prompt = prompt.tolist()
                    continue
                break
            return None
        
        prompts = []
        for idx, prompt in enumerate(prompts_raw):
            normalized = normalize_prompt(prompt)
            if normalized is None:
                print(f"⚠️  跳过未知格式的prompt[{idx}]: {type(prompt)}")
                continue
            prompts.append(normalized)
            if len(prompts) >= test_prompts_count:
                break
        
        if not prompts:
            print("❌ 未能加载任何有效的prompts")
            return False
        
        print(f"✓ 成功加载 {len(prompts)} 个prompts用于测试")
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建VERL workers
    print(f"\n创建 {num_workers} 个 VERL workers...")
    workers = []
    try:
        for i in range(num_workers):
            print(f"  初始化 Worker {i}...")
            os.environ["MASTER_PORT"] = str(29500 + i)
            worker = VERLRolloutWorker.remote(
                worker_id=i,
                model_path=model_path,
                max_tokens=max_tokens,
                gpu_memory=0.5,  # 小内存测试
                temperature=1.0,
            )
            workers.append(worker)
            print(f"  ✓ Worker {i} 创建成功")
        
        print(f"✓ 所有 {num_workers} 个workers创建完成")
        
    except Exception as e:
        print(f"❌ 创建workers失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试生成（模拟VERL默认调度：数据并行分割）
    print(f"\n开始测试生成（VERL默认调度方式）...")
    print(f"  将 {len(prompts)} 个prompts分配给 {num_workers} 个workers")
    
    # 数据并行分割
    prompts_per_worker = len(prompts) // num_workers
    prompt_batches = []
    for i in range(num_workers):
        start_idx = i * prompts_per_worker
        end_idx = (i + 1) * prompts_per_worker if i < num_workers - 1 else len(prompts)
        prompt_batches.append(prompts[start_idx:end_idx])
        print(f"  Worker {i}: {len(prompt_batches[i])} prompts")
    
    # 使用workers生成
    start_time = time.time()
    try:
        futures = []
        for i, worker in enumerate(workers):
            if prompt_batches[i]:
                print(f"  Worker {i} 开始生成...")
                future = worker.generate.remote(
                    prompts=prompt_batches[i],
                    n_samples=1,
                    temperature=1.0,
                )
                futures.append(future)
        
        # 等待所有workers完成
        print(f"\n等待所有workers完成...")
        results = ray.get(futures)
        end_time = time.time()
        
        print(f"✓ 生成完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 汇总结果
        all_responses = []
        for i, result in enumerate(results):
            if result and 'responses' in result:
                print(f"  Worker {i}: {len(result['responses'])} responses")
                all_responses.extend(result['responses'])
        
        print(f"\n✓ 总共生成 {len(all_responses)} 个responses")
        
        # 显示部分结果
        if all_responses:
            print(f"\n示例结果:")
            for i, resp in enumerate(all_responses[:2]):
                print(f"  Response {i+1}:")
                print(f"    Prompt length: {resp.get('prompt_length', 'N/A')}")
                print(f"    Response length: {resp.get('response_length', 'N/A')}")
                print(f"    Response text (前100字符): {resp.get('response_text', '')[:100]}...")
        
        print(f"\n{'='*70}")
        print(f"✅ 测试成功！")
        print(f"{'='*70}")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        print(f"\n清理资源...")
        try:
            ray.shutdown()
            print("✓ Ray已关闭")
        except:
            pass


if __name__ == "__main__":
    success = test_verl_default()
    sys.exit(0 if success else 1)

