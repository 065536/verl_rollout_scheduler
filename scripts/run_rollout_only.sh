#!/bin/bash

set -xeuo pipefail

# 项目根目录
PROJECT_ROOT=/data/250010176/codes/rollout_profiling_system
VERL_ROOT=${PROJECT_ROOT}/verl
MODEL_PATH=/data/250010176/codes/models/Qwen2.5-Math-7B

# 使用当前打开的数据集
DATA_FILE=${PROJECT_ROOT}/data/dapo_math_subset_128.parquet

# 目录设置 - 每次运行创建单独的日志文件夹
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${PROJECT_ROOT}/logs/rollout_${RUN_TIMESTAMP}
mkdir -p "$LOG_DIR"

# 设置日志输出到独立的日志文件夹
exec > >(tee -a "$LOG_DIR/rollout_only.log") 2>&1

echo "==========================================================="
echo "Log directory: $LOG_DIR"
echo "==========================================================="

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
# 16卡配置：可以通过环境变量覆盖，默认是2节点×8卡
# 单节点16卡：设置 NNODES=1 NGPUS_PER_NODE=16
# 多节点：设置 NNODES=2 NGPUS_PER_NODE=8
export NNODES=${NNODES:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# 设置PYTHONPATH，确保能找到verl包
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# 判断是否为HEAD节点
IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi

echo "HEAD_IP=$HEAD_IP IS_HEAD=$IS_HEAD RANK=${RANK:-unset} NODE_RANK=${NODE_RANK_CAND:-unset}"
echo "DATA_FILE=$DATA_FILE"
echo "MODEL_PATH=$MODEL_PATH"
echo "NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE (Total: $((NNODES * NGPUS_PER_NODE)) GPUs)"

# Ray管理
RAY=/data/250010176/yrh/miniconda3/envs/verl/bin/ray
$RAY stop -f || true
pkill -f redis-server || true
rm -rf /tmp/ray || true

if [ "$IS_HEAD" = "1" ]; then
  $RAY start --head --node-ip-address="$HEAD_IP" --port=6379 --dashboard-port=8265 --num-cpus=$(nproc) --num-gpus="${NGPUS_PER_NODE}"
  
  ready=0
  for i in $(seq 1 30); do
    if $RAY status >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  
  if [ "$ready" -ne 1 ]; then
    echo "Ray head failed to start"
    exit 1
  fi
  
  export RAY_ADDRESS="$HEAD_IP:6379"
  
  echo ""
  echo "==========================================================="
  echo "Applying per-worker metrics patch..."
  echo "==========================================================="
  # Apply patch to record per-worker rollout timing and token metrics
  cd "${PROJECT_ROOT}"
  python3 utils/patch_worker_metrics.py || {
    echo "Warning: Failed to apply patch, continuing without per-worker metrics"
  }
  
  echo ""
  echo "==========================================================="
  echo "Running Rollout Only (No Training)"
  echo "==========================================================="
  echo ""
  
  # 运行rollout only，使用verl包中的main_ppo
  # trainer.val_only=True 表示只运行验证（rollout），不进行训练
  # Set environment variable for log directory
  export VERL_LOG_DIR="${LOG_DIR}"
  
  cd "${VERL_ROOT}"
  python3 -m verl.trainer.main_ppo \
    data.train_files="${DATA_FILE}" \
    data.val_files="${DATA_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
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
    trainer.project_name="DAPO-Rollout-Only" \
    trainer.experiment_name="Qwen2.5-7b-MATH-Rollout-Only" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations=10 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${LOG_DIR}" \
    2>&1 | tee -a "$LOG_DIR/rollout_only.log"
  
  echo ""
  echo "Rollout completed successfully"
  $RAY stop || true
  exit 0
else
  $RAY start --address="$HEAD_IP:6379" --num-cpus=$(nproc)
  echo "Worker joined cluster"
  while $RAY status >/dev/null 2>&1; do sleep 10; done
  echo "Ray cluster stopped, worker exiting"
  exit 0
fi

