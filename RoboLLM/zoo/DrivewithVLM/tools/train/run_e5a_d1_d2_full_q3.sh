#!/usr/bin/env bash
# E5-A full training: D1 vehicle, then D2 infrastructure, both at q=3.
#
# Default layout (four physical 48 GB GPUs):
#   1) D1 vehicle: GPU 4,5,6,7 with DistributedDataParallel
#   2) D2 road:    GPU 4,5,6,7 with DistributedDataParallel
# Eight GPUs are also accepted via GPU_IDS=0,1,2,3,4,5,6,7.
#
# Data parallelism accelerates training; it does not pool VRAM.  Each rank
# still holds one complete three-frame sample.  D3 cooperative is excluded
# because its inference-only road-query path is not implemented yet.
#
# Run from any directory:
#   bash /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/tools/train/run_e5a_d1_d2_full_q3.sh
#
# Resume interrupted training (each stage resumes from its own latest.pth):
#   RESUME=1 bash /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/tools/train/run_e5a_d1_d2_full_q3.sh
#
# Optional overrides, e.g. EPOCHS=30 GPU_IDS=0,1,2,3,4,5,6,7 WORKERS=4 ...
# Eight-GPU example (use a distinct output directory):
#   GPU_IDS=0,1,2,3,4,5,6,7 RUN_ROOT=/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/work_dirs/e5a_d1_d2_q3_full_8gpu_e30 bash /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/tools/train/run_e5a_d1_d2_full_q3.sh

set -euo pipefail

UNIV2X_ROOT="/home/zzn/V2X_VLM/UniV2X"
DRIVE_ROOT="/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM"
PYTHON_BIN="/home/zzn/anaconda3/envs/univ2x/bin/python"

VEH_CONFIG="${DRIVE_ROOT}/projects/configs/Robodrivevlm/univ2x_e5a_vehicle_det.py"
INF_CONFIG="${DRIVE_ROOT}/projects/configs/Robodrivevlm/univ2x_e5a_infrastructure_det.py"

# Keep the current 40-epoch run intact; new 30-epoch runs use their own directory.
RUN_ROOT="${RUN_ROOT:-${DRIVE_ROOT}/work_dirs/e5a_d1_d2_q3_full_e30}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
MASTER_PORT="${MASTER_PORT:-29504}"
RESUME="${RESUME:-0}"

# On this server, the default cross-bridge PCIe P2P path hangs in NCCL
# broadcast (reproduced without a model on 2026-09-30). Keep NVLink P2P;
# NCCL uses shared memory between pairs. Allow explicit environment overrides.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${VEH_CONFIG}" || ! -f "${INF_CONFIG}" ]]; then
  echo "E5-A config missing under ${DRIVE_ROOT}" >&2
  exit 2
fi
if [[ "${NPROC}" -ne 4 && "${NPROC}" -ne 8 ]]; then
  echo "This training script supports four or eight GPUs; got GPU_IDS=${GPU_IDS}." >&2
  exit 2
fi

VEH_WORK_DIR="${RUN_ROOT}/vehicle"
INF_WORK_DIR="${RUN_ROOT}/infrastructure"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${VEH_WORK_DIR}" "${INF_WORK_DIR}" "${LOG_DIR}"

run_stage() {
  local name="$1"
  local config="$2"
  local work_dir="$3"
  local log_file="$4"
  local -a resume_args=()

  if [[ "${RESUME}" == "1" && -f "${work_dir}/latest.pth" ]]; then
    resume_args=(--resume-from "${work_dir}/latest.pth")
    echo "[$(date '+%F %T')] ${name}: resuming ${work_dir}/latest.pth"
  else
    echo "[$(date '+%F %T')] ${name}: starting from the BEVFormer initialization in config"
  fi

  echo "${name}: GPUs=${GPU_IDS}; world_size=${NPROC}; q=3; epochs=${EPOCHS}"
  echo "${name}: NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}; NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"
  (
    cd "${UNIV2X_ROOT}"
    PYTHONPATH="${UNIV2X_ROOT}" CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
      "${PYTHON_BIN}" -m torch.distributed.launch \
        --nproc_per_node="${NPROC}" \
        --master_addr=127.0.0.1 \
        --master_port="${MASTER_PORT}" \
        tools/train.py "${config}" \
        --launcher pytorch \
        --deterministic \
        --no-validate \
        --work-dir "${work_dir}" \
        "${resume_args[@]}" \
        --cfg-options \
          data.train.queue_length=3 \
          model_ego_agent.queue_length=3 \
          data.workers_per_gpu="${WORKERS}" \
          total_epochs="${EPOCHS}" \
          runner.max_epochs="${EPOCHS}" \
          checkpoint_config.interval="${CHECKPOINT_INTERVAL}" \
          log_config.interval="${LOG_INTERVAL}"
  ) 2>&1 | tee "${log_file}"
}

# Sequential on purpose: each stage receives all four GPUs.
run_stage vehicle "${VEH_CONFIG}" "${VEH_WORK_DIR}" "${LOG_DIR}/vehicle.log"
run_stage infrastructure "${INF_CONFIG}" "${INF_WORK_DIR}" "${LOG_DIR}/infrastructure.log"

echo "[$(date '+%F %T')] D1 and D2 full q=3 training completed."
echo "Checkpoints: ${VEH_WORK_DIR} and ${INF_WORK_DIR}"
