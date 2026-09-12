#!/usr/bin/env bash
# Pure rollout: use the training Gateway/task/reward path without a training engine.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
cd "${repo_root}"

DATA_DIR=${DATA_DIR:-/home/l00951262/input}
RUNTIME_DIR=${RUNTIME_DIR:-/home/l00951262/output}
MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3.5-27B"}
DATA_PATH=${DATA_PATH:-"${DATA_DIR}/data/uni_agent/eda_train.parquet"}
TASK_CONFIG=${TASK_CONFIG:-uni_agent/tasks/eda_agent/task_config_claude_code.yaml}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename -- "${MODEL_PATH}")"}
EXP_NAME=${EXP_NAME:-"$(date +%Y%m%d_%H%M%S)_rollout"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/EDA-rollout/${EXP_NAME}"}

# Export these values: Docker sandboxes are created by Ray task workers.
export EDA_DATASET_ROOT=${EDA_DATASET_ROOT:-/home/l00951262/EDA/dataset_innovus_19_10}
export EDA_SANDBOX_IMAGE=${EDA_SANDBOX_IMAGE:-crpi-lmega5fbvej4u3db.cn-shanghai.personal.cr.aliyuncs.com/novigrad/eda:v0.2-patch.2}
export EDA_SUBMISSION_DIR=${EDA_SUBMISSION_DIR:-"${AGENT_LOG_DIR}/submissions"}
export SANDBOX_STARTUP_CONCURRENCY=${SANDBOX_STARTUP_CONCURRENCY:-2}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${GEN_TP:-4}
CONCURRENCY=${CONCURRENCY:-2}
GATEWAY_COUNT=${GATEWAY_COUNT:-2}
LIMIT=${LIMIT:-4}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-32768}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-18000}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}

if ((NNODES <= 0 || NGPUS_PER_NODE <= 0 || ROLLOUT_TP <= 0)); then
    echo "Error: node counts, GPUs per node, and GEN_TP must be positive" >&2
    exit 2
fi
if (((NNODES * NGPUS_PER_NODE) % ROLLOUT_TP != 0)); then
    echo "Error: total GPUs must be divisible by GEN_TP" >&2
    exit 2
fi
if ((CONCURRENCY <= 0 || GATEWAY_COUNT <= 0 || LIMIT <= 0 || N_RESP_PER_PROMPT <= 0)); then
    echo "Error: concurrency, gateway count, sample limit, and rollout count must be positive" >&2
    exit 2
fi
for required_path in "${MODEL_PATH}/config.json" "${DATA_PATH}" "${TASK_CONFIG}" "${EDA_DATASET_ROOT}/tasks/index.tsv"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Error: required file does not exist: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p -- "${AGENT_LOG_DIR}" "${EDA_SUBMISSION_DIR}"
echo "Pure EDA rollout: model=${MODEL_PATH}, GPUs=${NNODES}x${NGPUS_PER_NODE}, TP=${ROLLOUT_TP}"
echo "Samples=${LIMIT}, attempts per sample=${N_RESP_PER_PROMPT}, concurrency=${CONCURRENCY}"
echo "Session logs and trajectories: ${AGENT_LOG_DIR}"
echo "Submitted repair.tcl files: ${EDA_SUBMISSION_DIR}"

python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --task-config "${TASK_CONFIG}" \
    --engine vllm \
    --tool-parser "${TOOL_PARSER}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${NGPUS_PER_NODE}" \
    --tensor-parallel-size "${ROLLOUT_TP}" \
    --gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --limit "${LIMIT}" \
    --n "${N_RESP_PER_PROMPT}" \
    --prompt-length "${MAX_PROMPT_LENGTH}" \
    --response-length "${MAX_RESPONSE_LENGTH}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --session-timeout-seconds "${SESSION_TIMEOUT_SECONDS}" \
    --enforce-eager \
    --log-task-results \
    --log-dir "${AGENT_LOG_DIR}" \
    --result-path "${AGENT_LOG_DIR}/result.json" \
    "$@" 2>&1 | tee "${AGENT_LOG_DIR}/driver.log"
