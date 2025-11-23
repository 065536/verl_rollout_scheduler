#!/bin/bash

set -xeuo pipefail

# 设置日志输出
exec > >(tee -a /data/250010176/codes/rollout_profiling_system/logs/dapo_training.log) 2>&1

# 项目根目录
PROJECT_ROOT=/data/250010176/codes/rollout_profiling_system
VERL_ROOT=${PROJECT_ROOT}/verl
DATA_ROOT=/data/250010176/data
MODEL_PATH=/data/250010176/codes/models/Qwen2.5-Math-7B

# 目录设置
CKPTS_DIR=${PROJECT_ROOT}/ckpts/DAPO/DAPO-Qwen2.5-7b-MATH-$(date +%Y%m%d)
LOG_DIR=${PROJECT_ROOT}/logs

mkdir -p "$CKPTS_DIR" "$LOG_DIR"

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
export TRAIN_FILE=${DATA_ROOT}/dapo-math-17k.parquet
export TEST_FILE=${DATA_ROOT}/aime-2024.parquet
export MODEL_PATH="$MODEL_PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NNODES=8
export NGPUS_PER_NODE=8
export CKPTS_DIR="$CKPTS_DIR"

# 设置PYTHONPATH，确保能找到verl包
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# 判断是否为HEAD节点
IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi

echo "HEAD_IP=$HEAD_IP IS_HEAD=$IS_HEAD RANK=${RANK:-unset} NODE_RANK=${NODE_RANK_CAND:-unset}"

# 检查和安装依赖
python - <<'PY'
import subprocess, sys

def pip_i(pkgs): subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", "--no-cache-dir"] + pkgs)

try:
    import typer
    v = tuple(int(x) for x in typer.__version__.split(".")[:2])
except Exception:
    v = None

if (v is None) or (v >= (0,12)):
    pip_i(["typer==0.9.0","click==8.1.7"])

try:
    import swanlab
    print(f"SwanLab already installed: {swanlab.__version__}")
except ImportError:
    print("Installing SwanLab...")
    pip_i(["swanlab"])
    print("SwanLab installed successfully")

print("Dependencies ready")
PY

# Ray管理
RAY=/data/250010176/yrh/miniconda3/envs/verl/bin/ray
$RAY stop -f || true
pkill -f redis-server || true
rm -rf /tmp/ray || true

if [ "$IS_HEAD" = "1" ]; then
  $RAY start --head --node-ip-address="$HEAD_IP" --port=6379 --dashboard-port=8265 --num-cpus=$(nproc)
  
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
  
  # 运行训练，使用verl包中的main_ppo
  # Hydra配置路径是相对于main_ppo.py的，需要在verl根目录运行
  # PYTHONPATH已设置，确保能找到verl包
  cd "${VERL_ROOT}"
  python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_batch_size=512 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=32 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=8192 \
    'trainer.logger=["console","swanlab"]' \
    trainer.project_name="DAPO" \
    trainer.experiment_name="DAPO-Qwen2.5-7b-MATH-SwanLab" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    2>&1 | tee -a "$LOG_DIR/dapo_7b_math_64gpu_swanlab.log"
  
  echo "Training completed successfully"
  $RAY stop || true
  exit 0
else
  $RAY start --address="$HEAD_IP:6379" --num-cpus=$(nproc)
  echo "Worker joined cluster"
  while $RAY status >/dev/null 2>&1; do sleep 10; done
  echo "Ray cluster stopped, worker exiting"
  exit 0
fi

