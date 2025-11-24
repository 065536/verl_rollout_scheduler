#!/bin/bash
# 批量运行三种调度模式的实验脚本
# 按顺序运行：verl官方（default）、shuffle调度、bin_packing调度
# 每种模式重复3次

set -xeuo pipefail

# 项目根目录
PROJECT_ROOT=/data/250010176/codes/rollout_profiling_system
VERL_ROOT=${PROJECT_ROOT}/verl
MODEL_PATH=/data/250010176/codes/models/Qwen3-4B

# 使用当前打开的数据集
DATA_FILE=${PROJECT_ROOT}/data/dapo_math_subset_128.parquet

# 从MODEL_PATH提取模型名称（去掉路径，只保留最后一部分）
MODEL_NAME=$(basename "$MODEL_PATH")

# 调度文件目录：schedules/{model_name}/
SCHEDULES_DIR="${PROJECT_ROOT}/schedules/${MODEL_NAME}"
mkdir -p "$SCHEDULES_DIR"

# 实验配置
REPEATS=3  # 每种调度模式重复3次
#SCHEDULE_MODES=("default" "shuffle" "bin_packing")  # 运行所有三种模式：default -> shuffle -> bin_packing
SCHEDULE_MODES=("bin_packing")  # 运行所有三种模式：default -> shuffle -> bin_packing

# 创建实验根目录
EXPERIMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ROOT=${PROJECT_ROOT}/logs/batch_scheduling_experiments_${EXPERIMENT_TIMESTAMP}
mkdir -p "$EXPERIMENT_ROOT"

# 创建实验日志文件
EXPERIMENT_LOG="${EXPERIMENT_ROOT}/batch_experiment.log"
exec > >(tee -a "$EXPERIMENT_LOG") 2>&1

echo "==========================================================="
echo "Batch Scheduling Experiments"
echo "==========================================================="
echo "Experiment Root: $EXPERIMENT_ROOT"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Schedule Modes: ${SCHEDULE_MODES[@]}"
echo "Repeats per mode: $REPEATS"
echo "==========================================================="
echo ""

# 获取HEAD IP
if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"
elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"
else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

# 激活conda环境
source /data/250010176/yrh/miniconda3/etc/profile.d/conda.sh
conda activate verl

cd "$PROJECT_ROOT"

# 设置环境变量
export MODEL_PATH="$MODEL_PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NNODES=${NNODES:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# 设置PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# 判断是否为HEAD节点
IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi

if [ "$IS_HEAD" = "0" ]; then
    # Worker节点：只启动Ray并等待
    RAY=/data/250010176/yrh/miniconda3/envs/verl/bin/ray
    
    # 清理旧的Ray进程
    $RAY stop -f || true
    pkill -9 -f "ray.*raylet" || true
    pkill -9 -f redis-server || true
    rm -rf /tmp/ray /tmp/plasma-* || true
    sleep 2
    
    # 连接到HEAD节点（显式指定GPU数量）
    echo "Worker node joining Ray cluster with ${NGPUS_PER_NODE} GPUs..."
    $RAY start --address="$HEAD_IP:6379" \
        --num-cpus=$(nproc) \
        --num-gpus="${NGPUS_PER_NODE}" \
        --object-store-memory=$((50 * 1024 * 1024 * 1024)) || {
        echo "ERROR: Worker failed to join Ray cluster"
        exit 1
    }
    
    echo "Worker joined cluster successfully, waiting for experiments to complete..."
    while $RAY status --address="$HEAD_IP:6379" >/dev/null 2>&1; do 
        sleep 10
    done
    echo "Ray cluster stopped, worker exiting"
    exit 0
fi

# HEAD节点：运行实验
RAY=/data/250010176/yrh/miniconda3/envs/verl/bin/ray

# 记录实验结果摘要
SUMMARY_FILE="${EXPERIMENT_ROOT}/experiment_summary.txt"
cat > "$SUMMARY_FILE" <<EOF
Batch Scheduling Experiments Summary
====================================
Start Time: $(date '+%Y-%m-%d %H:%M:%S')
Experiment Root: $EXPERIMENT_ROOT

Configuration:
  Model: $MODEL_PATH
  Data: $DATA_FILE
  Nodes: $NNODES
  GPUs per node: $NGPUS_PER_NODE
  Total GPUs: $((NNODES * NGPUS_PER_NODE))
  Schedule modes: ${SCHEDULE_MODES[@]}
  Repeats per mode: $REPEATS

Results:
EOF

# ============================================================
# 在实验开始前一次性启动Ray集群（所有实验共享）
# ============================================================
echo ""
echo "==========================================================="
echo "Starting Ray cluster (will be shared across all experiments)"
echo "==========================================================="

# 停止旧的Ray进程（如果有）
$RAY stop -f || true
pkill -9 -f "ray.*raylet" || true
pkill -9 -f redis-server || true
pkill -9 -f "ray.*dashboard" || true
rm -rf /tmp/ray /tmp/plasma-* || true
sleep 3

# 启动Ray HEAD节点（显式指定GPU数量）
echo "Starting Ray HEAD node with ${NGPUS_PER_NODE} GPUs per node..."
$RAY start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-port=8265 \
    --num-cpus=$(nproc) \
    --num-gpus="${NGPUS_PER_NODE}" \
    --object-store-memory=$((50 * 1024 * 1024 * 1024)) \
    || {
    echo "ERROR: Failed to start Ray head node"
    echo "EXPERIMENTS | FAILED: Ray head failed to start" >> "$SUMMARY_FILE"
    exit 1
}

# 等待Ray启动完成
ready=0
for i in $(seq 1 30); do
    if $RAY status >/dev/null 2>&1; then 
        ready=1
        break
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: Ray head failed to start within timeout"
    exit 1
fi

export RAY_ADDRESS="$HEAD_IP:6379"

# 等待所有节点连接并验证GPU资源
echo "Waiting for all ${NNODES} nodes to connect and verify GPU resources..."
TOTAL_REQUIRED_GPUS=$((NNODES * NGPUS_PER_NODE))

if [ "${NNODES}" -gt 1 ]; then
    echo "Waiting for all ${NNODES} nodes to connect..."
    nodes_ready=0
    for i in $(seq 1 90); do  # 增加等待时间到90次（4.5分钟）
        NUM_NODES=$($RAY status --address="$RAY_ADDRESS" 2>/dev/null | grep -c "node_" || echo 0)
        if [ "${NUM_NODES}" -ge "${NNODES}" ]; then
            nodes_ready=1
            echo "✓ All ${NNODES} nodes connected"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then  # 每30秒打印一次
            echo "  Progress: ${NUM_NODES}/${NNODES} nodes connected... (attempt $i/90)"
        fi
        sleep 3
    done
    
    if [ "$nodes_ready" -ne 1 ]; then
        echo "ERROR: Only ${NUM_NODES}/${NNODES} nodes connected. Cannot continue."
        echo "Please ensure Worker nodes are running and can connect to $HEAD_IP:6379"
        $RAY status --address="$RAY_ADDRESS" || true
        exit 1
    fi
fi

# 验证GPU资源是否足够
echo "Verifying GPU resources..."
sleep 3  # 等待节点完全注册
python3 <<EOF
import ray
import sys
import time

try:
    ray.init(address="$RAY_ADDRESS", ignore_reinit_error=True)
    
    # 等待一下让所有资源完全注册
    time.sleep(3)
    
    # 获取所有节点的GPU资源
    node_resources = ray._private.state.available_resources_per_node()
    total_gpus = 0
    for node_id, resources in node_resources.items():
        gpus = resources.get("GPU", 0.0)
        total_gpus += int(gpus)
        print(f"  Node {node_id[:16]}...: {int(gpus)} GPUs available")
    
    required_gpus = $TOTAL_REQUIRED_GPUS
    print(f"\nTotal GPUs detected: {total_gpus}")
    print(f"Total GPUs required: {required_gpus}")
    
    if total_gpus < required_gpus:
        print(f"\nERROR: Insufficient GPUs! Only {total_gpus}/{required_gpus} GPUs detected.")
        print("Ray cluster status:")
        import subprocess
        subprocess.run(["ray", "status", "--address", "$RAY_ADDRESS"], check=False)
        sys.exit(1)
    else:
        print(f"✓ GPU resources verified: {total_gpus} GPUs available")
        sys.exit(0)
        
except Exception as e:
    print(f"ERROR during GPU verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: GPU resource verification failed"
    $RAY status --address="$RAY_ADDRESS" || true
    exit 1
fi

echo "✓ Ray cluster ready for all experiments"
echo ""

# 遍历每种调度模式
for schedule_mode in "${SCHEDULE_MODES[@]}"; do
    echo ""
    echo "==========================================================="
    echo "Schedule Mode: $schedule_mode"
    echo "==========================================================="
    
    # 为每种调度模式创建子目录
    SCHEDULE_DIR="${EXPERIMENT_ROOT}/${schedule_mode}"
    mkdir -p "$SCHEDULE_DIR"
    
    # 遍历每次重复
    for repeat in $(seq 1 $REPEATS); do
        echo ""
        echo "-----------------------------------------------------------"
        echo "Schedule Mode: $schedule_mode | Repeat: $repeat/$REPEATS"
        echo "-----------------------------------------------------------"
        
        # 为每次重复创建子目录
        RUN_DIR="${SCHEDULE_DIR}/run_${repeat}"
        mkdir -p "$RUN_DIR"
        
        # 设置日志输出
        RUN_LOG="${RUN_DIR}/rollout.log"
        
        # 设置调度模式环境变量
        export VERL_SCHEDULE_MODE="$schedule_mode"
        export VERL_LOG_DIR="${RUN_DIR}"
        
        # 根据调度模式设置相应的环境变量
        if [ "$schedule_mode" = "shuffle" ]; then
            # 启用shuffle调度
            export VERL_ENABLE_SHUFFLE="true"
            # 使用repeat编号作为随机种子，确保每次重复有不同的随机性
            export VERL_SHUFFLE_SEED=$((repeat * 1000))
            echo "Using shuffle scheduling (seed: $VERL_SHUFFLE_SEED)"
        else
            # 禁用shuffle（如果是default或bin_packing模式）
            export VERL_ENABLE_SHUFFLE="false"
            unset VERL_SHUFFLE_SEED
        fi
        
        # 如果使用bin_packing，设置调度文件路径（需要基于default实验结果生成）
        if [ "$schedule_mode" = "bin_packing" ]; then
            # 计算worker数量和replica数量
            NUM_WORKERS=$((NNODES * NGPUS_PER_NODE))
            TP_SIZE=2  # 从配置中读取tensor_model_parallel_size
            NUM_REPLICAS=$((NUM_WORKERS / TP_SIZE))
            
            # 调度文件命名格式：bin_packing_{model_name}_{num_replicas}r.json
            # 例如：bin_packing_Qwen3-4B_8r.json
            SCHEDULE_FILE="${SCHEDULES_DIR}/bin_packing_${MODEL_NAME}_${NUM_REPLICAS}r.json"
            
            # 首先检查是否存在预先生成的调度文件
            if [ -f "$SCHEDULE_FILE" ]; then
                export VERL_BIN_PACKING_SCHEDULE_FILE="$SCHEDULE_FILE"
                echo "Using existing bin packing schedule file: $SCHEDULE_FILE"
            else
                # 优先从当前实验的default结果中查找metrics文件（按照实验顺序：default -> shuffle -> bin_packing）
                # default实验必须已经完成，才能为bin_packing生成调度文件
                DEFAULT_METRICS_FILE=""
                DEFAULT_METRICS_DIR="${SCHEDULE_DIR}/../default"
                
                if [ -d "$DEFAULT_METRICS_DIR" ]; then
                    # 查找default模式中最近一次运行的metrics文件
                    # 优先查找本次实验中的default结果（所有run）
                    DEFAULT_METRICS_FILE=$(find "$DEFAULT_METRICS_DIR" -name "rollout_worker_metrics_default_*.json" -type f | sort | tail -1)
                    if [ -n "$DEFAULT_METRICS_FILE" ] && [ -f "$DEFAULT_METRICS_FILE" ]; then
                        echo "Found metrics file from default experiment in current batch: $DEFAULT_METRICS_FILE"
                    fi
                fi
                
                # 如果在当前实验中没找到，尝试从其他实验目录查找default结果
                if [ -z "$DEFAULT_METRICS_FILE" ] || [ ! -f "$DEFAULT_METRICS_FILE" ]; then
                    # 查找logs目录下最近完成的default实验的metrics文件
                    FOUND_DEFAULT=$(find "${PROJECT_ROOT}/logs" -path "*/default/*" -name "rollout_worker_metrics_default_*.json" -type f | sort | tail -1)
                    if [ -n "$FOUND_DEFAULT" ] && [ -f "$FOUND_DEFAULT" ]; then
                        echo "Found metrics file from previous default experiment: $FOUND_DEFAULT"
                        DEFAULT_METRICS_FILE="$FOUND_DEFAULT"
                    fi
                fi
                
                # 如果找到了default的metrics文件，使用它生成bin packing调度文件
                if [ -n "$DEFAULT_METRICS_FILE" ] && [ -f "$DEFAULT_METRICS_FILE" ]; then
                    echo "Generating bin packing schedule from default experiment metrics..."
                    echo "  Metrics file: $DEFAULT_METRICS_FILE"
                    echo "  This will calculate avg response lengths from default experiment results"
                    echo "  and use Best Fit Decreasing algorithm to create bin packing schedule"
                    
                    mkdir -p "$SCHEDULES_DIR"
                    
                    # 运行schedule_prompts.py生成调度文件
                    # schedule_prompts.py会从metrics文件中读取all_per_prompt_avg_response_lengths
                    # 然后使用Best Fit Decreasing算法生成bin packing调度
                    if python3 "${PROJECT_ROOT}/utils/schedule_prompts.py" \
                        "$DEFAULT_METRICS_FILE" \
                        --num-replicas "$NUM_REPLICAS" \
                        --output "$SCHEDULE_FILE" 2>&1; then
                        if [ -f "$SCHEDULE_FILE" ]; then
                            export VERL_BIN_PACKING_SCHEDULE_FILE="$SCHEDULE_FILE"
                            echo "✓ Successfully generated bin packing schedule file from default experiment: $SCHEDULE_FILE"
                        else
                            echo "ERROR: Schedule file generation completed but file not found: $SCHEDULE_FILE"
                        fi
                    else
                        echo "ERROR: Failed to generate bin packing schedule from metrics file"
                        echo "  Check if the metrics file contains 'all_per_prompt_avg_response_lengths' data"
                    fi
                else
                    # 如果找不到default的metrics文件，给出错误并跳过
                    echo "ERROR: Cannot find default experiment results!"
                    echo "  Bin packing requires default experiment to be run first."
                    echo "  Expected location: ${DEFAULT_METRICS_DIR}/run_*/rollout_worker_metrics_default_*.json"
                    echo "  The experiment order must be: default -> shuffle -> bin_packing"
                    echo "  Skipping this bin_packing experiment."
                    echo "$schedule_mode | run_${repeat} | FAILED: Default experiment results not found" >> "$SUMMARY_FILE"
                    continue
                fi
                
                # 如果还是没有调度文件，给出错误并跳过这个实验
                if [ -z "${VERL_BIN_PACKING_SCHEDULE_FILE:-}" ] || [ ! -f "${VERL_BIN_PACKING_SCHEDULE_FILE:-}" ]; then
                    echo "ERROR: Bin packing schedule file not generated!"
                    echo "  Expected location: ${SCHEDULE_FILE}"
                    echo "  Bin packing mode requires a schedule file to work."
                    echo "  Skipping this experiment."
                    echo "$schedule_mode | run_${repeat} | FAILED: Bin packing schedule file not generated" >> "$SUMMARY_FILE"
                    continue
                fi
                
                # 验证调度文件是否有效
                if ! python3 -c "import json; json.load(open('${VERL_BIN_PACKING_SCHEDULE_FILE}'))" 2>/dev/null; then
                    echo "ERROR: Bin packing schedule file is invalid or corrupted: ${VERL_BIN_PACKING_SCHEDULE_FILE}"
                    echo "$schedule_mode | run_${repeat} | FAILED: Invalid bin packing schedule file" >> "$SUMMARY_FILE"
                    continue
                fi
                
                echo "✓ Bin packing schedule file validated: ${VERL_BIN_PACKING_SCHEDULE_FILE}"
            fi
        fi
        
        # 注意：shuffle的环境变量已经在上面设置了
        
        # 应用patch（如果需要）
        echo "Applying patches..."
        cd "${PROJECT_ROOT}"
        python3 utils/patch_worker_metrics.py || {
            echo "Warning: Failed to apply patch, continuing without per-worker metrics"
        }
        
        # 记录开始时间
        RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        RUN_START_SECONDS=$(date +%s)
        
        # 运行rollout only实验
        echo "Running rollout experiment..."
        cd "${VERL_ROOT}"
        
        python3 -m verl.trainer.main_ppo \
            data.train_files="${DATA_FILE}" \
            data.val_files="${DATA_FILE}" \
            data.prompt_key=prompt \
            data.truncation='left' \
            data.max_prompt_length=2048 \
            data.max_response_length=32768 \
            data.train_batch_size=128 \
            actor_rollout_ref.rollout.n=16 \
            actor_rollout_ref.actor.strategy=fsdp \
            actor_rollout_ref.actor.ppo_mini_batch_size=64 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.rollout.temperature=0.6 \
            actor_rollout_ref.rollout.top_p=1.0 \
            actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.rollout.val_kwargs.n=1 \
            critic.enable=False \
            critic.strategy=fsdp \
            critic.ppo_micro_batch_size_per_gpu=32 \
            reward_model.reward_manager=dapo \
            +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
            +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
            +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
            +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
            +reward_model.reward_kwargs.max_resp_len=8192 \
            'trainer.logger=["console"]' \
            trainer.project_name="Batch-Scheduling-Experiments" \
            trainer.experiment_name="${schedule_mode}-run${repeat}" \
            trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
            trainer.nnodes="${NNODES}" \
            trainer.val_before_train=True \
            trainer.val_only=True \
            trainer.log_val_generations=10 \
            trainer.save_freq=-1 \
            trainer.total_epochs=1 \
            trainer.default_local_dir="${RUN_DIR}" \
            2>&1 | tee -a "$RUN_LOG"
        
        RUN_EXIT_CODE=$?
        RUN_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        RUN_END_SECONDS=$(date +%s)
        RUN_DURATION=$((RUN_END_SECONDS - RUN_START_SECONDS))
        
        # 不在这里停止Ray，让所有实验共享同一个Ray集群
        # 所有实验结束后会统一清理
        
        # 检查是否生成了metrics文件
        METRICS_FILE=$(find "${RUN_DIR}" -name "rollout_worker_metrics_${schedule_mode}_*.json" | head -1)
        if [ -n "$METRICS_FILE" ] && [ "$RUN_EXIT_CODE" -eq 0 ]; then
            echo "SUCCESS: Run completed successfully"
            echo "Metrics file: $METRICS_FILE"
            echo "Duration: ${RUN_DURATION} seconds"
            echo "$schedule_mode | run_${repeat} | SUCCESS | Duration: ${RUN_DURATION}s | Metrics: $(basename $METRICS_FILE)" >> "$SUMMARY_FILE"
        else
            echo "FAILED: Run failed or metrics file not found"
            echo "$schedule_mode | run_${repeat} | FAILED | Exit code: $RUN_EXIT_CODE" >> "$SUMMARY_FILE"
        fi
        
        # 等待一下再继续下一个实验（让系统清理资源）
        sleep 5
    done
    
    echo ""
    echo "Completed all repeats for schedule mode: $schedule_mode"
done

# 生成最终摘要
echo "" >> "$SUMMARY_FILE"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "Total Duration: $(($(date +%s) - $(date -d "$(head -n 5 "$SUMMARY_FILE" | grep "Start Time" | cut -d: -f2-)" +%s)))s" >> "$SUMMARY_FILE"

echo ""
echo "==========================================================="
echo "All Experiments Completed"
echo "==========================================================="
echo "Experiment Root: $EXPERIMENT_ROOT"
echo "Summary: $SUMMARY_FILE"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 显示实验结构
echo "Experiment Structure:"
tree -L 2 "$EXPERIMENT_ROOT" || find "$EXPERIMENT_ROOT" -type d -maxdepth 2 | sort

# 停止Ray（最后清理）
$RAY stop || true

echo ""
echo "Batch experiments completed successfully!"

