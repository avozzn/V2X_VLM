#!/usr/bin/env bash
# User-launched test evaluation; checkpoints selected on validation only.
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zzn/anaconda3/envs/robollm/bin/python}"
RUN_ROOT="${RUN_ROOT:-work_dirs/T1_S1_objects_4gpu_smoke_20261004}"
CACHE="${CACHE:-data/Planning/dual_object_a1_v1_20261004}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
EVAL_ROOT="${EVAL_ROOT:-evaluation_results/T1_S1_test_$(date +%Y%m%d_%H%M%S)}"
trap 'echo "评测失败，日志目录：$EVAL_ROOT" >&2' ERR
"$PYTHON_BIN" - "$EVAL_GPUS" <<'CHECK'
import sys
ids=sys.argv[1].split(',')
assert len(ids) in (1,2,4) and len(set(ids))==len(ids) and all(x.isdigit() for x in ids), 'Use 1,2 or4 distinct GPU indices'
CHECK
IFS=',' read -r -a DEVICE_IDS <<< "$EVAL_GPUS"
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" -u -m tools.object_vlm.evaluate freeze --run-root "$RUN_ROOT" --cache "$CACHE" --output-dir "$EVAL_ROOT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "检查通过，仅冻结checkpoint；未启动GPU。结果目录：$EVAL_ROOT"
  exit 0
fi
for label in T1 S1; do
  printf '开始 %s test评测：GPUs=%s，共480帧\n' "$label" "$EVAL_GPUS"
  CUDA_VISIBLE_DEVICES="$EVAL_GPUS" "$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${#DEVICE_IDS[@]}" -m tools.object_vlm.evaluate predict \
    --experiment "$label" --selection "$EVAL_ROOT/frozen_selection.json" \
    --output-dir "$EVAL_ROOT/$label" 2>&1 | tee "$EVAL_ROOT/${label}.log"
done
"$PYTHON_BIN" -u -m tools.object_vlm.evaluate summarize --output-dir "$EVAL_ROOT"
printf '评测完成：%s/report.md\n' "$EVAL_ROOT"
