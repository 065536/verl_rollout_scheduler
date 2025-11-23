#!/bin/bash
# 通用的Rollout Profiling启动脚本（包含Ray管理）
# 支持三种调度方式：task_scheduler, bin_packing, verl_default
set -xeuo pipefail

ROOT=/data/250010176/codes/rollout_profiling_system
ENV_ROOT=/data/250010176/yrh/miniconda3
ENV_NAME=verl

# 使用传递的实验目录，如果没有则使用默认目录
if [ -n "${EXP_OUTPUT_DIR:-}" ]; then
    LOG_DIR="${EXP_OUTPUT_DIR}"
    echo "使用指定的实验目录: ${LOG_DIR}"
else
    # 独立运行时，创建带时间戳的实验目录
    ts=$(date +"%Y%m%d_%H%M%S")
    SCHEDULER_NAME="${SCHEDULER:-task_scheduler}"
    LOG_DIR="${ROOT}/profiling_results/rollout_${SCHEDULER_NAME}_${ts}"
    echo "创建新的实验目录: ${LOG_DIR}"
fi

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/profiling.log"
exec &> >(tee -a "${LOG_FILE}")

echo "==========================================================="
echo "Rollout Profiling with Scheduler: ${SCHEDULER:-task_scheduler}"
echo "==========================================================="

# 获取HEAD IP
if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"
elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"
else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

echo "HEAD_IP: ${HEAD_IP}"

# 激活环境
source "${ENV_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT}"

# 环境变量
export PYTHONPATH="${ROOT}:${ROOT}/verl:${PYTHONPATH:-}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
if [ -z "${NUM_WORKERS+x}" ] || [ -z "${NUM_WORKERS:-}" ]; then
    export NUM_WORKERS=$(( NNODES * NGPUS_PER_NODE ))
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRUST_REMOTE_CODE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_FORCE_NON_NVML=1
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_PRECOMPILED=0

# 缓存目录（vLLM/torch.compile）
mkdir -p /root/.cache/vllm/torch_compile_cache
mkdir -p /root/.cache/vilm/torch_compile_cache
mkdir -p ~/.cache/torch
chmod -R 777 /root/.cache/ 2>/dev/null || true

# 配置参数（可通过环境变量覆盖）
MODEL_PATH="${MODEL_PATH:-/data/250010176/codes/models/Qwen3-4B}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/dapo_math_subset_128.parquet}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
GPU_MEMORY="${GPU_MEMORY:-0.8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:--1}"
TOP_P="${TOP_P:-1.0}"
SCHEDULER="${SCHEDULER:-task_scheduler}"
WORKER_TYPE="${WORKER_TYPE:-vllm}"
REMAINING_ROUNDS="${REMAINING_ROUNDS:-1}"
SCHEDULE_CSV="${SCHEDULE_CSV:-}"

echo ""
echo "Profiling Configuration:"
echo "  Scheduler: ${SCHEDULER}"
echo "  Worker type: ${WORKER_TYPE}"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_FILE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-k: ${TOP_K}"
echo "  Top-p: ${TOP_P}"
echo "  GPU memory: ${GPU_MEMORY}"
echo "  Nodes: ${NNODES}"
echo "  GPUs per node: ${NGPUS_PER_NODE}"
echo "  Total GPUs: $((NNODES * NGPUS_PER_NODE))"
echo "  Output dir: ${LOG_DIR}"
echo ""

# 保存配置信息
cat > "${LOG_DIR}/experiment_config.txt" <<EOF
Rollout Profiling Configuration
========================================
Start Time: $(date '+%Y-%m-%d %H:%M:%S')

Scheduler: ${SCHEDULER}
Worker Type: ${WORKER_TYPE}

Model Configuration:
  Model path: ${MODEL_PATH}
  Data file: ${DATA_FILE}
  
Generation Parameters:
  Max tokens: ${MAX_TOKENS}
  Temperature: ${TEMPERATURE}
  Top-k: ${TOP_K}
  Top-p: ${TOP_P}
  
Worker Configuration:
  Num workers: ${NUM_WORKERS}
  GPU memory fraction: ${GPU_MEMORY}
  Remaining rounds: ${REMAINING_ROUNDS}
  
Cluster Configuration:
  Nodes: ${NNODES}
  GPUs per node: ${NGPUS_PER_NODE}
  Total GPUs: $((NNODES * NGPUS_PER_NODE))
  
Output:
  Log directory: ${LOG_DIR}
  Log file: ${LOG_FILE}
EOF

# Ray管理
RAY="${ENV_ROOT}/envs/${ENV_NAME}/bin/ray"
$RAY stop -f || true
pkill -f redis-server || true
rm -rf /tmp/ray || true
sleep 2

NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-${RANK:-}}}"
NODE_RANK="${NODE_RANK_CAND:-0}"
IS_HEAD=0
[ "${NODE_RANK}" = "0" ] && IS_HEAD=1

echo "NODE_RANK=${NODE_RANK} IS_HEAD=${IS_HEAD}"

if [ "$IS_HEAD" = "1" ]; then
    echo ""
    echo "==========================================================="
    echo "Starting Ray HEAD node..."
    echo "==========================================================="
    
    $RAY start --head --node-ip-address="${HEAD_IP}" --port=6379 --dashboard-port=8265 \
        --num-cpus="$(nproc)" --num-gpus="${NGPUS_PER_NODE}" \
        --object-store-memory=$((50 * 1024 * 1024 * 1024))
    
    # 等待Ray启动
    for i in $(seq 1 40); do 
        $RAY status >/dev/null 2>&1 && break
        sleep 2
    done
    
    export RAY_ADDRESS="${HEAD_IP}:6379"
    
    echo "OK Ray HEAD node started"
    echo "  Dashboard: http://${HEAD_IP}:8265"
    echo ""
    
    # 等待worker节点连接
    if [ "${NNODES}" -gt 1 ]; then
        echo "Waiting for ${NNODES} nodes to connect..."
        for i in $(seq 1 60); do
            NUM_NODES=$($RAY status | grep -c "node_" || echo 0)
            if [ "${NUM_NODES}" -ge "${NNODES}" ]; then
                echo "OK All ${NNODES} nodes connected"
                break
            fi
            echo "  Currently ${NUM_NODES}/${NNODES} nodes connected..."
            sleep 5
        done
        sleep 5
    fi
    
    # 显示Ray cluster状态
    echo ""
    echo "==========================================================="
    echo "Ray Cluster Status:"
    echo "==========================================================="
    $RAY status
    echo ""
    
    # 运行Rollout Profiling
    echo ""
    echo "==========================================================="
    echo "Starting Rollout Profiling..."
    echo "==========================================================="
    echo ""
    
    # 构建命令
    CMD="python3 ${ROOT}/run_rollout_profiling.py \
        --scheduler ${SCHEDULER} \
        --worker_type ${WORKER_TYPE} \
        --model_path ${MODEL_PATH} \
        --num_workers ${NUM_WORKERS} \
        --max_tokens ${MAX_TOKENS} \
        --dataset ${DATA_FILE} \
        --temperature ${TEMPERATURE} \
        --top_k ${TOP_K} \
        --top_p ${TOP_P} \
        --gpu_memory ${GPU_MEMORY} \
        --remaining_rounds ${REMAINING_ROUNDS} \
        --output_dir ${LOG_DIR}"
    
    # 如果使用bin_packing且有CSV文件，添加参数
    if [ "${SCHEDULER}" = "bin_packing" ]; then
        if [ -n "${SCHEDULE_CSV}" ]; then
            CMD="${CMD} --schedule_csv ${SCHEDULE_CSV}"
        elif [ -f "${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv" ]; then
            CMD="${CMD} --schedule_csv ${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv"
            echo "✓ 自动使用调度文件: bin_packing_schedule_${NUM_WORKERS}w.csv"
        fi
    fi
    
    # 执行命令
    eval ${CMD}
    
    PROFILING_EXIT_CODE=$?
    
    echo ""
    if [ ${PROFILING_EXIT_CODE} -eq 0 ]; then
        echo "SUCCESS Profiling completed successfully!"
    else
        echo "ERROR Profiling failed with exit code: ${PROFILING_EXIT_CODE}"
    fi
    
    # 显示生成的文件
    echo ""
    echo "==========================================================="
    echo "Generated Files:"
    echo "==========================================================="
    ls -lh "${LOG_DIR}"/ 2>/dev/null || echo "No files found"
    echo ""
    
    # 停止Ray
    $RAY stop || true
    
    # 添加结束时间
    echo "" >> "${LOG_DIR}/experiment_config.txt"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_DIR}/experiment_config.txt"
    
else
    # Worker节点
    echo ""
    echo "==========================================================="
    echo "Starting Ray WORKER node..."
    echo "==========================================================="
    
    $RAY start --address="${HEAD_IP}:6379" \
        --num-cpus="$(nproc)" --num-gpus="${NGPUS_PER_NODE}" \
        --object-store-memory=$((50 * 1024 * 1024 * 1024))
    
    echo "OK Ray worker node started, connected to ${HEAD_IP}:6379"
    
    # 等待head节点的任务完成
    while $RAY status >/dev/null 2>&1; do 
        sleep 10
    done
fi

echo ""
echo "Script completed at $(date)"
echo "Log: ${LOG_FILE}"
echo ""

