#!/usr/bin/env python3
"""
通用的Rollout Profiling启动脚本
支持三种调度方式：
1. task_scheduler - 实时动态调度（最大优先策略）
2. bin_packing - 静态预分配调度（Bin Packing算法）
3. verl_default - VERL框架默认调度（数据并行dispatch）
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import ray

# 确保在导入前设置环境变量
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl"))

from transformers import AutoTokenizer

from rollout_profiling.workers.verl_worker import VERLRolloutWorker
from rollout_profiling.workers.vllm_worker import VLLMRolloutWorker
from rollout_profiling.utils.ema_predictor import EMALengthPredictor
from rollout_profiling.utils.scheduler import TaskScheduler
from rollout_profiling.utils.bin_packing import BinPackingScheduler
from rollout_profiling.utils.verl_utils import (
    prompts_to_dataproto,
    dataproto_to_responses,
    create_verl_rollout_config,
    setup_verl_environment,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def compute_basic_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {
            "mean": float(values[0]),
            "std": 0.0,
            "min": float(values[0]),
            "max": float(values[0]),
        }
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def normalize_prompt(prompt: Any) -> List[Dict[str, Any]] | None:
    """将各种格式的prompt转换为消息列表"""
    import pandas as pd  # local import to avoid issues when pandas absent

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


def load_prompts_from_parquet(parquet_path: str, prompt_key: str = "prompt") -> List[Any]:
    """从parquet文件加载prompts"""
    print(f"📂 加载数据集: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✓ 数据集大小: {len(df)} 条")
    
    # 提取prompts
    if prompt_key in df.columns:
        prompts = df[prompt_key].tolist()
    else:
        # 尝试查找可能的prompt列
        possible_keys = ["messages", "input", "query", "question"]
        for key in possible_keys:
            if key in df.columns:
                print(f"⚠️  未找到'{prompt_key}'列，使用'{key}'列")
                prompts = df[key].tolist()
                break
        else:
            raise ValueError(f"未找到prompt列，可用列: {df.columns.tolist()}")
    
    # 处理prompts格式
    processed_prompts = []
    for i, prompt in enumerate(prompts):
        normalized = normalize_prompt(prompt)
        if normalized is None:
            print(f"⚠️  Prompt {i} 格式未知: {type(prompt)}, 跳过")
            continue
        processed_prompts.append(normalized)
    
    print(f"✓ 成功加载 {len(processed_prompts)} 个prompts")
    return processed_prompts


def extract_worker_rows(scheduler_name: str, results: Dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if scheduler_name == "task_scheduler":
        worker_stats = results.get('worker_stats', {})
        for worker_id, stats in worker_stats.items():
            rows.append({
                'worker_id': worker_id,
                'num_responses': stats.get('num_responses', 0),
                'total_tokens': stats.get('total_tokens', 0),
                'total_time': stats.get('total_time', 0.0),
                'pure_generation_time': stats.get('pure_generation_time', 0.0),
            })
    else:
        worker_results = results.get('worker_results', [])
        for entry in worker_results:
            responses = entry.get('responses', [])
            total_tokens = sum(resp.get('response_length', 0) for resp in responses)
            rows.append({
                'worker_id': entry.get('worker_id'),
                'num_responses': entry.get('num_responses', len(responses)),
                'total_tokens': total_tokens,
                'total_time': entry.get('worker_duration'),
                'pure_generation_time': entry.get('pure_generation_duration'),
            })
    return rows


def run_task_scheduler_mode(
    prompts: List[Any],
    num_workers: int,
    worker_type: str,
    model_path: str,
    max_tokens: int,
    remaining_rounds: int = 1,
    **kwargs
):
    """使用TaskScheduler进行实时动态调度"""
    print(f"\n{'='*70}")
    print(f"🚀 启动 TaskScheduler 模式（实时动态调度）")
    print(f"{'='*70}")
    
    # 初始化EMA预测器
    num_prompts = len(prompts)
    ema_predictor = EMALengthPredictor(
        num_prompts=num_prompts,
        initial_length=512.0,
        alpha=0.3
    )
    
    # 创建TaskScheduler
    ema_state = ema_predictor.export_state()
    scheduler = TaskScheduler.remote(
        ema_state=ema_state,
        remaining_rounds=remaining_rounds,
    )
    
    # 创建workers
    print(f"\n创建 {num_workers} 个 {worker_type} workers...")
    workers = []
    for i in range(num_workers):
        if worker_type == "verl":
            worker = VERLRolloutWorker.remote(
                worker_id=i,
                model_path=model_path,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            worker = VLLMRolloutWorker.remote(
                worker_id=i,
                model_path=model_path,
                max_tokens=max_tokens,
                **kwargs
            )
        workers.append(worker)
    
    # 启动worker处理任务
    print(f"\n开始处理任务...")
    start_time = time.time()
    
    futures = []
    for worker in workers:
        future = worker.process_task_queue.remote(
            scheduler=scheduler,
            prompts=prompts,
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", -1),
            top_p=kwargs.get("top_p", 1.0),
        )
        futures.append(future)
    
    # 等待所有worker完成
    results = ray.get(futures)
    end_time = time.time()
    
    # 获取最终结果
    final_results = ray.get(scheduler.get_all_results.remote())
    
    print(f"\n{'='*70}")
    print(f"✅ TaskScheduler 模式完成")
    print(f"{'='*70}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"处理的任务数: {sum(r['processed_tasks'] for r in results)}")
    print(f"总响应数: {len(final_results['responses'])}")
    
    return final_results


def run_bin_packing_mode(
    prompts: List[Any],
    num_workers: int,
    worker_type: str,
    model_path: str,
    max_tokens: int,
    schedule_csv: str = None,
    **kwargs
):
    """使用BinPackingScheduler进行静态预分配调度"""
    print(f"\n{'='*70}")
    print(f"🚀 启动 BinPacking 模式（静态预分配调度）")
    print(f"{'='*70}")
    
    # 初始化EMA预测器（用于预测长度）
    num_prompts = len(prompts)
    ema_predictor = EMALengthPredictor(
        num_prompts=num_prompts,
        initial_length=512.0,
        alpha=0.3
    )
    
    # 获取预测长度
    prompt_lengths = ema_predictor.get_all_expected_lengths()
    
    # 如果提供了CSV文件，从CSV加载调度计划
    if schedule_csv and os.path.exists(schedule_csv):
        print(f"📋 从CSV文件加载调度计划: {schedule_csv}")
        df = pd.read_csv(schedule_csv)
        worker_assignments = []
        for i in range(num_workers):
            worker_row = df[df['worker_id'] == i]
            if not worker_row.empty:
                import ast
                prompt_ids = ast.literal_eval(worker_row.iloc[0]['prompt_ids'])
                worker_assignments.append(prompt_ids)
            else:
                worker_assignments.append([])
        print(f"✓ 从CSV加载了 {num_workers} 个worker的分配计划")
    else:
        # 使用BinPackingScheduler生成调度计划
        print(f"📋 使用BinPackingScheduler生成调度计划...")
        bin_packer = BinPackingScheduler(num_workers=num_workers)
        worker_assignments = bin_packer.schedule_prompts(prompt_lengths)
    
    # 创建workers
    print(f"\n创建 {num_workers} 个 {worker_type} workers...")
    workers = []
    for i in range(num_workers):
        if worker_type == "verl":
            worker = VERLRolloutWorker.remote(
                worker_id=i,
                model_path=model_path,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            worker = VLLMRolloutWorker.remote(
                worker_id=i,
                model_path=model_path,
                max_tokens=max_tokens,
                **kwargs
            )
        workers.append(worker)
    
    # 为每个worker分配prompts并生成
    print(f"\n开始处理任务...")
    start_time = time.time()
    
    futures = []
    for i, worker in enumerate(workers):
        assigned_prompt_ids = worker_assignments[i]
        if assigned_prompt_ids:
            assigned_prompts = [prompts[pid] for pid in assigned_prompt_ids]
            future = worker.generate.remote(
                prompts=assigned_prompts,
                n_samples=1,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", -1),
                top_p=kwargs.get("top_p", 1.0),
            )
            futures.append(future)
        else:
            print(f"⚠️  Worker {i} 没有分配任何prompts")
    
    # 等待所有worker完成
    results = ray.get(futures)
    end_time = time.time()
    
    # 汇总结果
    all_responses = []
    for result in results:
        if result and 'responses' in result:
            all_responses.extend(result['responses'])
    
    print(f"\n{'='*70}")
    print(f"✅ BinPacking 模式完成")
    print(f"{'='*70}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"总响应数: {len(all_responses)}")
    
    return {
        'responses': all_responses,
        'worker_results': results,
    }


def _build_process_layout(total_workers: int, nnodes: int, gpus_per_node: int) -> List[int]:
    """计算RayResourcePool的 process_on_nodes 布局"""
    remaining = total_workers
    layout: List[int] = []
    for _ in range(max(nnodes, 1)):
        if remaining <= 0:
            break
        assign = min(gpus_per_node, remaining) if gpus_per_node > 0 else remaining
        layout.append(assign)
        remaining -= assign
    while remaining > 0:
        assign = min(gpus_per_node, remaining) if gpus_per_node > 0 else remaining
        layout.append(assign)
        remaining -= assign
    return [c for c in layout if c > 0]


def run_verl_default_mode(
    prompts: List[Any],
    num_workers: int,
    model_path: str,
    max_tokens: int,
    nnodes: int = 1,
    gpus_per_node: int = 1,
    **kwargs,
):
    """使用VERL官方Ray WorkerGroup调度"""
    print(f"\n{'='*70}")
    print(f"🚀 启动 VERL默认调度 模式（Ray WorkerGroup + generate_sequences）")
    print(f"{'='*70}")

    setup_verl_environment()
    total_world_size = num_workers
    layout = _build_process_layout(total_world_size, nnodes, gpus_per_node)
    if sum(layout) != total_world_size:
        raise ValueError(
            f"资源布局异常: 期望世界尺寸 {total_world_size}, 实际 {sum(layout)} (layout={layout})"
        )
    print(f"Ray资源布局（process_on_nodes）: {layout}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = create_verl_rollout_config(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=kwargs.get("temperature", 1.0),
        top_k=kwargs.get("top_k", -1),
        top_p=kwargs.get("top_p", 1.0),
        gpu_memory=kwargs.get("gpu_memory", 0.5),
    )
    config.actor.fsdp_config["fsdp_size"] = total_world_size
    config.rollout["data_parallel_size"] = total_world_size

    ray_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config,
        role="rollout",
    )
    resource_pool = RayResourcePool(process_on_nodes=layout, use_gpu=True, max_colocate_count=1)
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls,
        device_name="cuda",
    )
    print("⏳ 初始化VERL WorkerGroup模型...")
    worker_group.init_model()

    dataproto = prompts_to_dataproto(prompts, tokenizer)
    attention_mask = dataproto.batch["attention_mask"]
    original_prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]

    padded_proto, pad_size = pad_dataproto_to_divisor(dataproto, worker_group.world_size)
    print(f"发送 DataProto（{len(dataproto)} 条，padding={pad_size}）到 WorkerGroup")

    start_time = time.time()
    output_padded = worker_group.generate_sequences(padded_proto)
    output = unpad_dataproto(output_padded, pad_size=pad_size)
    total_duration = time.time() - start_time

    timing_info = output.meta_info.get("timing", {})
    responses = dataproto_to_responses(
        dataproto=output,
        tokenizer=tokenizer,
        prompt_ids=list(range(len(prompts))),
        original_prompt_lengths=original_prompt_lengths,
    )
    total_tokens = sum(resp.get("response_length", 0) for resp in responses)

    print(f"\n{'='*70}")
    print("✅ VERL默认调度 模式完成")
    print(f"{'='*70}")
    print(f"总耗时: {total_duration:.2f} 秒 | 响应条数: {len(responses)}")
    if timing_info:
        print(f"VERL返回的生成耗时统计: {timing_info}")

    worker_results = [
        {
            "worker_id": "verl_dp_group",
            "num_responses": len(responses),
            "total_tokens": total_tokens,
            "worker_duration": total_duration,
            "pure_generation_duration": timing_info.get("generation_timing/mean", total_duration),
            "timing_info": timing_info,
        }
    ]

    return {
        "responses": responses,
        "worker_results": worker_results,
        "timing": timing_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Rollout Profiling启动脚本")
    
    # 基本参数
    parser.add_argument("--scheduler", type=str, required=True,
                        choices=["task_scheduler", "bin_packing", "verl_default"],
                        help="调度方式: task_scheduler, bin_packing, verl_default")
    parser.add_argument("--worker_type", type=str, default="vllm",
                        choices=["verl", "vllm"],
                        help="Worker类型: verl 或 vllm")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Worker数量")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="最大生成token数")
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "1")),
                        help="Ray集群节点数（默认读取NNODES环境变量）")
    parser.add_argument("--gpus_per_node", type=int, default=int(os.environ.get("NGPUS_PER_NODE", "1")),
                        help="每个节点可用GPU数量（默认读取NGPUS_PER_NODE环境变量）")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str,
                        default="data/dapo_math_subset_128.parquet",
                        help="数据集parquet文件路径")
    parser.add_argument("--prompt_key", type=str, default="prompt",
                        help="prompt列名")
    
    # BinPacking特定参数
    parser.add_argument("--schedule_csv", type=str, default=None,
                        help="BinPacking调度计划CSV文件（可选）")
    
    # TaskScheduler特定参数
    parser.add_argument("--remaining_rounds", type=int, default=1,
                        help="每个prompt生成的轮数（TaskScheduler模式）")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k采样")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus采样")
    parser.add_argument("--gpu_memory", type=float, default=0.5,
                        help="GPU内存使用率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="profiling_results",
                        help="结果输出目录")
    
    args = parser.parse_args()
    
    # 初始化Ray
    if not ray.is_initialized():
        init_kwargs = {"ignore_reinit_error": True}
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            init_kwargs["address"] = ray_address
            print(f"连接现有Ray集群: {ray_address}")
        else:
            init_kwargs["num_cpus"] = max(args.num_workers + 2, 2)
        ray.init(**init_kwargs)
        print(f"✓ Ray初始化完成")
    
    # 加载数据集
    prompts = load_prompts_from_parquet(args.dataset, args.prompt_key)
    
    # 准备kwargs
    kwargs = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "gpu_memory": args.gpu_memory,
    }
    
    overall_start_time = time.time()

    # 根据调度方式运行
    if args.scheduler == "task_scheduler":
        results = run_task_scheduler_mode(
            prompts=prompts,
            num_workers=args.num_workers,
            worker_type=args.worker_type,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            remaining_rounds=args.remaining_rounds,
            **kwargs
        )
    elif args.scheduler == "bin_packing":
        results = run_bin_packing_mode(
            prompts=prompts,
            num_workers=args.num_workers,
            worker_type=args.worker_type,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            schedule_csv=args.schedule_csv,
            **kwargs
        )
    elif args.scheduler == "verl_default":
        results = run_verl_default_mode(
            prompts=prompts,
            num_workers=args.num_workers,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            nnodes=args.nnodes,
            gpus_per_node=args.gpus_per_node,
            **kwargs
        )
    else:
        raise ValueError(f"未知的调度方式: {args.scheduler}")
    
    overall_duration = time.time() - overall_start_time

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"profiling_{args.scheduler}_{args.worker_type}_{timestamp}"
    responses_csv = os.path.join(args.output_dir, f"{base_name}_responses.csv")
    workers_csv = os.path.join(args.output_dir, f"{base_name}_workers.csv")
    summary_json = os.path.join(args.output_dir, f"{base_name}_summary.json")
    
    responses = results.get('responses', [])
    responses_df = pd.DataFrame(responses) if responses else pd.DataFrame()
    if not responses_df.empty:
        responses_df.to_csv(
            responses_csv,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
    else:
        responses_csv = None
    
    worker_rows = extract_worker_rows(args.scheduler, results)
    worker_df = pd.DataFrame(worker_rows) if worker_rows else pd.DataFrame()
    if not worker_df.empty:
        worker_df.to_csv(
            workers_csv,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
    else:
        workers_csv = None

    response_length_stats = {}
    if not responses_df.empty and 'response_length' in responses_df.columns:
        lengths = responses_df['response_length'].dropna().astype(float).tolist()
        response_length_stats = compute_basic_stats(lengths)

    worker_timing_stats = {}
    if not worker_df.empty and 'total_time' in worker_df.columns:
        timings = worker_df['total_time'].dropna().astype(float).tolist()
        worker_timing_stats = compute_basic_stats(timings)
        worker_timing_stats["num_workers"] = len(timings)

    summary = {
        "scheduler": args.scheduler,
        "worker_type": args.worker_type,
        "model_path": args.model_path,
        "num_workers": args.num_workers,
        "num_prompts": len(prompts),
        "total_responses": len(responses),
        "total_time_sec": overall_duration,
        "response_length": response_length_stats,
        "worker_timing": worker_timing_stats,
        "files": {
            "responses_csv": responses_csv,
            "workers_csv": workers_csv,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 输出文件:")
    if responses_csv:
        print(f"  - Responses: {responses_csv}")
    if workers_csv:
        print(f"  - Workers:   {workers_csv}")
    print(f"  - Summary:   {summary_json}")
    print(f"\n🎉 完成！总耗时 {overall_duration:.2f} 秒")


if __name__ == "__main__":
    main()

