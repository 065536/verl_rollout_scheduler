#!/bin/bash
# 批量运行不同调度方案的实验
# 支持三种调度方式：task_scheduler, bin_packing, verl_default
# 支持重复次数配置
set -euo pipefail

ROOT=/data/250010176/codes/rollout_profiling_system
ENV_ROOT=/data/250010176/yrh/miniconda3
ENV_NAME=verl

# 激活环境
source "${ENV_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${ROOT}"

# 获取HEAD IP（所有节点都需要）
if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"
elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"
else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

echo "HEAD_IP: ${HEAD_IP}"

# 环境变量对齐
export PYTHONPATH="${ROOT}:${ROOT}/verl:${PYTHONPATH:-}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
if [ -z "${NUM_WORKERS:-}" ]; then
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

# 实验配置（可通过环境变量覆盖）
MODEL_PATH="${MODEL_PATH:-/data/250010176/codes/models/Qwen3-4B}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/dapo_math_subset_128.parquet}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
GPU_MEMORY="${GPU_MEMORY:-0.8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:--1}"
TOP_P="${TOP_P:-1.0}"
WORKER_TYPE="${WORKER_TYPE:-vllm}"

# 调度方案配置（可通过环境变量覆盖）
# 格式：SCHEDULERS="task_scheduler bin_packing verl_default"
SCHEDULERS="${SCHEDULERS:-task_scheduler bin_packing}"

# 重复次数（可通过环境变量覆盖）
REPEAT_TIMES="${REPEAT_TIMES:-3}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志入口
if [ -z "${BATCH_DIR:-}" ]; then
    BATCH_DIR="/tmp"
fi
if [ -z "${MASTER_LOG:-}" ]; then
    MASTER_LOG="/tmp/batch_experiments.log"
fi

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
    echo -e "${BLUE}实验配置：${NC}"
    echo "  Model: ${MODEL_PATH}"
    echo "  Data: ${DATA_FILE}"
    echo "  Workers: ${NUM_WORKERS}"
    echo "  Worker type: ${WORKER_TYPE}"
    echo "  Max tokens: ${MAX_TOKENS}"
    echo "  Temperature: ${TEMPERATURE}"
    echo "  Top-k: ${TOP_K}"
    echo "  Top-p: ${TOP_P}"
    echo "  GPU memory: ${GPU_MEMORY}"
    echo ""
    echo -e "${BLUE}调度方案：${NC}"
    for sched in ${SCHEDULERS}; do
        echo "  - ${sched}"
    done
    echo ""
    echo -e "${BLUE}重复次数：${REPEAT_TIMES}${NC}"
    echo ""
    # Head节点日志/目录
    BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BATCH_DIR="${ROOT}/profiling_results/batch_${BATCH_TIMESTAMP}"
    mkdir -p "${BATCH_DIR}"
    MASTER_LOG="${BATCH_DIR}/batch_experiments.log"
    exec > >(tee -a "${MASTER_LOG}") 2>&1

    echo "==========================================================="
    echo "批量Rollout Profiling实验"
    echo "==========================================================="
    echo "批量实验目录: ${BATCH_DIR}"
    echo "总日志文件: ${MASTER_LOG}"
    echo ""

    echo -e "${YELLOW}即将开始批量实验...${NC}"
    sleep 2
else
    BATCH_DIR="/tmp"
    # worker节点仅记录本地日志
    MASTER_LOG="/tmp/worker_${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${MASTER_LOG}") 2>&1
    echo "Worker node log: ${MASTER_LOG}"
    BATCH_DIR=""
fi

# 函数：运行单个实验
run_experiment() {
    local scheduler=$1
    local repeat_num=$2
    local total_repeats=$3
    
    local exp_timestamp=$(date +%Y%m%d_%H%M%S)
    local exp_name="${scheduler}_repeat${repeat_num}_${exp_timestamp}"
    local exp_dir="${BATCH_DIR}/${exp_name}"
    mkdir -p "${exp_dir}"
    
    # 保存实验配置
if [ "$IS_HEAD" = "1" ]; then
    cat > "${exp_dir}/config.txt" <<EOF
Experiment: ${repeat_num}/${total_repeats} for ${scheduler}
Scheduler: ${scheduler}
Repeat: ${repeat_num}/${total_repeats}
Start time: $(date '+%Y-%m-%d %H:%M:%S')

Configuration:
  Model: ${MODEL_PATH}
  Data: ${DATA_FILE}
  Workers: ${NUM_WORKERS}
  Worker type: ${WORKER_TYPE}
  Max tokens: ${MAX_TOKENS}
  Temperature: ${TEMPERATURE}
  Top-k: ${TOP_K}
  Top-p: ${TOP_P}
  GPU memory: ${GPU_MEMORY}
EOF
fi
    
    echo ""
    echo "========================================================================"
    echo -e "${CYAN}实验: ${scheduler} (${repeat_num}/${total_repeats})${NC}"
    echo "========================================================================"
    echo "  实验目录: ${exp_dir}"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 设置环境变量
    export SCHEDULER="${scheduler}"
    export WORKER_TYPE="${WORKER_TYPE}"
    export MODEL_PATH="${MODEL_PATH}"
    export DATA_FILE="${DATA_FILE}"
    export NUM_WORKERS="${NUM_WORKERS}"
    export MAX_TOKENS="${MAX_TOKENS}"
    export TEMPERATURE="${TEMPERATURE}"
    export TOP_K="${TOP_K}"
    export TOP_P="${TOP_P}"
    export GPU_MEMORY="${GPU_MEMORY}"
    export REMAINING_ROUNDS="${REMAINING_ROUNDS:-1}"
    export EXP_OUTPUT_DIR="${exp_dir}"  # 传递实验目录给子脚本
    
    # 如果使用bin_packing，尝试自动查找CSV文件
    if [ "${scheduler}" = "bin_packing" ]; then
        if [ -f "${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv" ]; then
            export SCHEDULE_CSV="${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv"
            echo "✓ 使用调度文件: bin_packing_schedule_${NUM_WORKERS}w.csv"
        elif [ -n "${SCHEDULE_CSV:-}" ]; then
            echo "✓ 使用指定的调度文件: ${SCHEDULE_CSV}"
        else
            echo -e "${YELLOW}⚠️  未找到 ${NUM_WORKERS} workers 的调度文件，将自动生成${NC}"
        fi
    else
        unset SCHEDULE_CSV || true
    fi
    
    # 构建命令
    CMD="python3 ${ROOT}/run_rollout_profiling.py \
        --scheduler ${scheduler} \
        --worker_type ${WORKER_TYPE} \
        --model_path ${MODEL_PATH} \
        --num_workers ${NUM_WORKERS} \
        --max_tokens ${MAX_TOKENS} \
        --dataset ${DATA_FILE} \
        --temperature ${TEMPERATURE} \
        --top_k ${TOP_K} \
        --top_p ${TOP_P} \
        --gpu_memory ${GPU_MEMORY} \
        --remaining_rounds ${REMAINING_ROUNDS:-1} \
        --output_dir ${exp_dir}"
    
    # 如果使用bin packing，尝试自动查找CSV文件
    if [ "${scheduler}" = "bin_packing" ]; then
        if [ -f "${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv" ]; then
            CMD="${CMD} --schedule_csv ${ROOT}/data/scheduling_plans/bin_packing_schedule_${NUM_WORKERS}w.csv"
            echo "✓ 使用调度文件: bin_packing_schedule_${NUM_WORKERS}w.csv"
        elif [ -n "${SCHEDULE_CSV:-}" ]; then
            CMD="${CMD} --schedule_csv ${SCHEDULE_CSV}"
            echo "✓ 使用指定的调度文件: ${SCHEDULE_CSV}"
        else
            echo -e "${YELLOW}⚠️  未找到 ${NUM_WORKERS} workers 的调度文件，将自动生成${NC}"
        fi
    fi
    
    # 运行实验
    if eval ${CMD}; then
        echo ""
        echo -e "${GREEN}✅ 实验完成: ${scheduler} (${repeat_num}/${total_repeats})${NC}"
        echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${exp_dir}/config.txt"
        echo "Status: SUCCESS" >> "${exp_dir}/config.txt"
        return 0
    else
        echo ""
        echo -e "${RED}❌ 实验失败: ${scheduler} (${repeat_num}/${total_repeats})${NC}"
        echo "  失败时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Status: FAILED at $(date '+%Y-%m-%d %H:%M:%S')" >> "${exp_dir}/config.txt"
        return 1
    fi
}

# 统计变量
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

if [ "$IS_HEAD" = "1" ]; then
    # 启动Ray head
    echo ""
    echo "==========================================================="
    echo "Starting Ray HEAD node..."
    echo "==========================================================="
    $RAY start --head --node-ip-address="${HEAD_IP}" --port=6379 --dashboard-port=8265 \
        --num-cpus="$(nproc)" --num-gpus="${NGPUS_PER_NODE}" \
        --object-store-memory=$((50 * 1024 * 1024 * 1024))

    for i in $(seq 1 40); do 
        $RAY status >/dev/null 2>&1 && break
        sleep 2
    done

    export RAY_ADDRESS="${HEAD_IP}:6379"

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

    # 运行所有实验
    for scheduler in ${SCHEDULERS}; do
        echo ""
        echo "========================================================================"
        echo -e "${BLUE}开始调度方案: ${scheduler}${NC}"
        echo "========================================================================"
        echo ""
        
        for repeat in $(seq 1 ${REPEAT_TIMES}); do
            TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
            
            if run_experiment "${scheduler}" "${repeat}" "${REPEAT_TIMES}"; then
                SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
            else
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
            fi
            
            if [ "${repeat}" -lt "${REPEAT_TIMES}" ] || [ "${scheduler}" != "$(echo ${SCHEDULERS} | awk '{print $NF}')" ]; then
                echo ""
                echo "等待30秒让系统稳定..."
                sleep 30
            fi
        done
    done

    $RAY stop || true

    # 完成统计
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}批量实验完成${NC}"
    echo "========================================================================"
    echo ""
    echo "统计信息:"
    echo "  总实验数: ${TOTAL_EXPERIMENTS}"
    echo -e "  ${GREEN}成功: ${SUCCESSFUL_EXPERIMENTS}${NC}"
    if [ ${FAILED_EXPERIMENTS} -gt 0 ]; then
        echo -e "  ${RED}失败: ${FAILED_EXPERIMENTS}${NC}"
    else
        echo "  失败: ${FAILED_EXPERIMENTS}"
    fi
    echo ""
    echo "批量实验目录: ${BATCH_DIR}"
    echo ""

    cat >> "${BATCH_DIR}/batch_config.txt" <<EOF

实验结果:
  总实验数: ${TOTAL_EXPERIMENTS}
  成功: ${SUCCESSFUL_EXPERIMENTS}
  失败: ${FAILED_EXPERIMENTS}
结束时间: $(date '+%Y-%m-%d %H:%M:%S')
EOF

    echo "实验结果目录结构："
    ls -lh "${BATCH_DIR}" | head -20
    echo ""

    echo "========================================================================"
    echo -e "${BLUE}实验摘要${NC}"
    echo "========================================================================"
    echo ""
    for scheduler in ${SCHEDULERS}; do
        echo -e "${CYAN}${scheduler}:${NC}"
        scheduler_dirs=$(find "${BATCH_DIR}" -maxdepth 1 -type d -name "${scheduler}_repeat*" | sort)
        if [ -n "${scheduler_dirs}" ]; then
            for exp_dir in ${scheduler_dirs}; do
                exp_name=$(basename "${exp_dir}")
                if [ -f "${exp_dir}/config.txt" ]; then
                    status=$(grep "^Status:" "${exp_dir}/config.txt" | awk '{print $2}' || echo "UNKNOWN")
                    if [ "${status}" = "SUCCESS" ]; then
                        echo -e "  ${GREEN}✓${NC} ${exp_name}"
                    else
                        echo -e "  ${RED}✗${NC} ${exp_name}"
                    fi
                else
                    echo -e "  ${YELLOW}?${NC} ${exp_name}"
                fi
            done
        else
            echo -e "  ${YELLOW}未找到实验结果${NC}"
        fi
        echo ""
    done

    echo "========================================================================"
    echo ""
    echo "下一步："
    echo "  1. 查看各实验目录下的结果文件"
    echo "  2. 对比不同调度方案的效果"
    echo "  3. 分析实验结果"
    echo ""
    echo "实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "==========================================================="
    echo "Starting Ray WORKER node..."
    echo "==========================================================="
    $RAY start --address="${HEAD_IP}:6379" \
        --num-cpus="$(nproc)" --num-gpus="${NGPUS_PER_NODE}" \
        --object-store-memory=$((50 * 1024 * 1024 * 1024))

    echo "OK Ray worker node started, connected to ${HEAD_IP}:6379"

    while $RAY status >/dev/null 2>&1; do 
        sleep 10
    done

    $RAY stop || true
fi

