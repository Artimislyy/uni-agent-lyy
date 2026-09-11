#!/usr/bin/env bash
set -euo pipefail

# 单机 NCCL 初始化崩溃的排查配置：绕过外部网络插件和 IB，输出详细日志。
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 从任意目录调用都切回仓库根目录，保证相对路径一致。
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${repo_root}"

# 默认使用 A100 服务器的路径和镜像；环境变量非空时优先使用环境变量。
DATA_DIR=${DATA_DIR:-/home/l00951262/input}
RUNTIME_DIR=${RUNTIME_DIR:-/home/l00951262/output}
EDA_DATASET_ROOT=${EDA_DATASET_ROOT:-/home/l00951262/EDA/dataset_innovus_19_10}
EDA_SANDBOX_IMAGE=${EDA_SANDBOX_IMAGE:-crpi-lmega5fbvej4u3db.cn-shanghai.personal.cr.aliyuncs.com/novigrad/eda:v0.2-patch.2}

project_name=${PROJECT_NAME:-Uni-Agent-EDA-Qwen3.5-4B-megatron}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/eda_train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/eda_validation.parquet"}
TASK_CONFIG=${TASK_CONFIG:-uni_agent/tasks/eda_agent/task_config_claude_code.yaml}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
SUBMISSION_DIR=${SUBMISSION_DIR:-"${RUNTIME_DIR}/submissions/${project_name}/${exp_name}"}

mkdir -p -- "${SUBMISSION_DIR}"

# 模型训练卡与 rollout 卡是两个不重叠的 Ray resource pool。
NNODES_TRAIN=${NNODES_TRAIN:-1}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TRAIN_TP=${TP:-2}
TRAIN_PP=${PP:-1}
TRAIN_CP=${CP:-2}
ROLLOUT_TP=${GEN_TP:-2}

# EDA episode 很慢且消耗 Innovus license，默认并发比 SWE-bench 保守。
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-4} # 每步训练 XX 个题目
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2} #每次更新用 XX 个
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-2} #每 XX 次更新同步一次权重给推理引擎
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4} #每道题做 XX 条轨迹
CONCURRENCY=${CONCURRENCY:-2} #最多同时运行**个session
GATEWAY_COUNT=${GATEWAY_COUNT:-2} #启动XX个gateway actor
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-18000} #一个session最长运行 5 小时（18000 秒）
SANDBOX_STARTUP_CONCURRENCY=${SANDBOX_STARTUP_CONCURRENCY:-16} #限制“同时启动多少个 sandbox”。

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((24 * 1024))}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((128 * 1024))}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"} # Agent 发请求时用的就是这个别名
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-True} #没做完的轨迹要不要参与训练，True 会 mask 掉，False 会直接参与训练。True 更安全，False 更快。

USE_MBRIDGE=${USE_MBRIDGE:-False} # Megatron（NVIDIA 的训练框架）内部为了并行训练，会把权重切分、重排成自己专属的排布方式，和 HF 格式完全不同，mBridge 负责转换
USE_DIST_CKPT=${USE_DIST_CKPT:-False}
OFFLOAD=${OFFLOAD:-True}
OFFLOAD_FRACTION=${OFFLOAD_FRACTION:-1.0} #优化器状态 offload 的比例
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7} #vLLM 推理引擎最多占用 GPU 显存的 **%
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-0} #正式训练前先空跑几个 batch“热身”（不更新参数，只让各组件跑通、显存分配稳定）。0 = 不热身，直接开训。
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10} #整个训练数据集完整过 ** 遍
SAVE_FREQ=${SAVE_FREQ:-10} #每 ** 个训练步保存一次 checkpoint
LR_DECAY_STEPS=${LR_DECAY_STEPS:-2000} #学习率衰减计划的步数

if ((TRAIN_PROMPT_BSZ != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE)); then
    echo "Error: TRAIN_PROMPT_BSZ must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi
if ((NNODES_TRAIN <= 0 || NNODES_ROLLOUT <= 0 || NGPUS_PER_NODE <= 0)); then
    echo "Error: training nodes, rollout nodes, and GPUs per node must all be greater than 0" >&2
    exit 2
fi
if ((TRAIN_TP <= 0 || TRAIN_PP <= 0 || TRAIN_CP <= 0 || ROLLOUT_TP <= 0)); then
    echo "Error: TP, PP, CP, and ROLLOUT_TP must all be greater than 0" >&2
    exit 2
fi
train_world_size=$((NNODES_TRAIN * NGPUS_PER_NODE))
train_model_parallel_size=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))
if ((train_world_size % train_model_parallel_size != 0)); then
    echo "Error: the total number of training GPUs must be divisible by TP * PP * CP" >&2
    exit 2
fi
if (((NNODES_ROLLOUT * NGPUS_PER_NODE) % ROLLOUT_TP != 0)); then
    echo "Error: the total number of rollout GPUs must be divisible by ROLLOUT_TP" >&2
    exit 2
fi

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Error: required path does not exist: ${required_path}" >&2
        exit 2
    fi
done
if [[ ! -f "${EDA_DATASET_ROOT}/tasks/index.tsv" ]]; then
    echo "Error: tasks/index.tsv was not found under EDA_DATASET_ROOT" >&2
    exit 2
fi

# checkpoint 若仍是 Git LFS 指针，上传到 sandbox 后 Innovus 必然无法恢复。
# 检查Parquet 是否一个“完整、自包含”的数据集。
python3 -m uni_agent.tasks.eda_agent.preprocess \
    --dataset-root "${EDA_DATASET_ROOT}" \
    --check-only \
    --require-lfs-materialized \
    --preview 0

total_context=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) #一条样本的最大总长度
actor_token_len=$((total_context / TRAIN_CP)) #摊到单卡上的长度
infer_token_len=$((total_context / TRAIN_CP)) 

# 数组让每个 Hydra override 都保持一个独立参数，便于审查和追加 "$@" 覆盖。
trainer_args=(
    --config-name=ppo_megatron_trainer
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}
    trainer.v1.separate_async.hybrid_rollout.enable_switch=False
    trainer.v1.sampler.max_off_policy_threshold=8
    trainer.v1.sampler.max_off_policy_strategy=drop
    transfer_queue.enable=True

    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation=error
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_PROMPT_BSZ}

    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=False
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean

    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${total_context}
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.lr_decay_steps=${LR_DECAY_STEPS}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT}
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP}
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    "+actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model']"

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes=${NNODES_ROLLOUT}
    actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_token_len}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_num_batched_tokens=${total_context}
    actor_rollout_ref.rollout.max_model_len=${total_context}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}
    actor_rollout_ref.rollout.agent.num_workers=8

    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=${SESSION_TIMEOUT_SECONDS}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False

    actor_rollout_ref.nccl_timeout=9600
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_token_len}
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT}
    actor_rollout_ref.ref.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP}

    reward.reward_manager.name=dapo
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    "trainer.logger=['console']"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=${SAVE_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    # 默认不恢复 TransferQueue，避免旧的 in-flight 请求在训练卡休眠前被重新下发。
    # 需要续训时可在脚本末尾追加 trainer.resume_mode=auto，但会放宽严格不共卡保证。
    trainer.resume_mode=disable
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.nnodes=${NNODES_TRAIN}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}

    # 直接启动 Python 后，verl 内部仍使用 Ray；显式传入 worker 环境变量。
    "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET_PLUGIN='${NCCL_NET_PLUGIN}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET='${NCCL_NET}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG='${NCCL_DEBUG}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.EDA_DATASET_ROOT='${EDA_DATASET_ROOT}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.EDA_SANDBOX_IMAGE='${EDA_SANDBOX_IMAGE}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.EDA_SUBMISSION_DIR='${SUBMISSION_DIR}'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_STARTUP_CONCURRENCY='${SANDBOX_STARTUP_CONCURRENCY}'"
)

echo "Starting EDA separate_async: train=${NNODES_TRAIN}x${NGPUS_PER_NODE} GPUs, rollout=${NNODES_ROLLOUT}x${NGPUS_PER_NODE} GPUs"
echo "Each step uses ${TRAIN_PROMPT_BSZ} prompts x ${N_RESP_PER_PROMPT} trajectories; sandbox concurrency limit: ${CONCURRENCY}"
echo "Accepted repair.tcl files will be saved under ${SUBMISSION_DIR}"

# Docker 由 ray_task 所在宿主机的 daemon 创建，各候选节点需要 Docker 权限和同名镜像。
python3 -m verl.trainer.main_ppo \
    "${trainer_args[@]}" \
    "$@"
