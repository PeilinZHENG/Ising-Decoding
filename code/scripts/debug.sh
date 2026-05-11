#!/usr/bin/env bash
set -euo pipefail

export EXPERIMENT_NAME=ddp8_1h_e200

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="${REPO_ROOT}/code"

cd "${REPO_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ddp8_1h_e200}"
CONFIG_NAME="${CONFIG_NAME:-config_public}"
WORKFLOW="${WORKFLOW:-train}"

BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
LOG_BASE_DIR="${PREDECODER_LOG_BASE_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_BASE_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${EXPERIMENT_NAME}_${TIMESTAMP}"
LOG_DIR="${LOG_BASE_DIR}/${RUN_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export PREDECODER_TRAIN_EPOCHS=200
export PREDECODER_TRAIN_SAMPLES=32768
export PREDECODER_VAL_SAMPLES=4096
export PREDECODER_TEST_SAMPLES=4096
export PREDECODER_DISABLE_SDR=1
export PREDECODER_LER_FINAL_ONLY=1
export PREDECODER_EVAL_NUM_WORKERS=0
export PREDECODER_SDR_NUM_WORKERS=0
export PREDECODER_INFERENCE_NUM_WORKERS=0
export PREDECODER_TORCH_COMPILE=0

export CUSTOM_DIST_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Debug run (8 GPU, 1h target)" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "workflow.task: ${WORKFLOW}" | tee -a "${LOG_FILE}"
echo "config: ${CONFIG_NAME}" | tee -a "${LOG_FILE}"
echo "GPUS: 8 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" | tee -a "${LOG_FILE}"
echo "output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "logs: ${LOG_DIR}" | tee -a "${LOG_FILE}"
echo "train/val/test samples: ${PREDECODER_TRAIN_SAMPLES}/${PREDECODER_VAL_SAMPLES}/${PREDECODER_TEST_SAMPLES}" | tee -a "${LOG_FILE}"
echo "epochs: ${PREDECODER_TRAIN_EPOCHS}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

{
  echo "[System] date=$(date)"
  echo "[System] host=$(hostname)"
  echo "[System] python=$(python -V 2>&1)"
  echo "[System] pytorch=$(python - <<'PY'
import torch
print(f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY
)"
  echo "[System] nvidia-smi (name,driver,memory)"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} | tee -a "${LOG_FILE}"

python -u -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=1 \
  code/workflows/run.py \
  --config-name="${CONFIG_NAME}" \
  workflow.task="${WORKFLOW}" \
  +exp_tag="${EXPERIMENT_NAME}" \
  ++load_checkpoint=False \
  +job_time_limit_seconds=3600 \
  hydra.run.dir="${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"

cp -f "${LOG_FILE}" "${OUTPUT_DIR}/run.log"
echo "Done. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
