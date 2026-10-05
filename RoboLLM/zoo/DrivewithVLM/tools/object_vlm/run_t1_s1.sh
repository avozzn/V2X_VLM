#!/usr/bin/env bash
# User-launched experiments only. Never evicts or stops another GPU task.
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zzn/anaconda3/envs/robollm/bin/python}"
T1_GPU="${T1_GPU:-0}"
S1_GPU="${S1_GPU:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
ACTION="${ACTION:-all}" # all | smoke | train; train requires prior smoke outputs
RUN_ROOT="${RUN_ROOT:-work_dirs/T1_S1_objects_$(date +%Y%m%d_%H%M%S)}"
CACHE="${CACHE:-data/Planning/dual_object_a1_v1_20261004}"
trap 'echo "脚本失败，请检查日志目录：$RUN_ROOT" >&2' ERR
if [[ "$T1_GPU" == "$S1_GPU" ]]; then
  echo 'T1_GPU and S1_GPU must be different physical devices.' >&2
  exit 2
fi
if [[ "$ACTION" != all && "$ACTION" != smoke && "$ACTION" != train ]]; then
  echo 'ACTION must be all, smoke or train.' >&2
  exit 2
fi
if [[ "$ACTION" == train ]]; then
  for run_label in T1 S1; do
    if [[ ! -f "$RUN_ROOT/${run_label}_smoke/smoke.json" ]]; then
      echo "Missing successful $run_label smoke in $RUN_ROOT. Run ACTION=smoke first." >&2
      exit 2
    fi
  done
else
  mkdir "$RUN_ROOT"
fi
IFS=',' read -r -a TRAIN_DEVICE_IDS <<< "$TRAIN_GPUS"
if [[ "${#TRAIN_DEVICE_IDS[@]}" != 4 ]]; then
  echo 'TRAIN_GPUS must list exactly four distinct physical devices.' >&2
  exit 2
fi
"$PYTHON_BIN" - "$TRAIN_GPUS" <<'DEVICES'
import sys
ids=sys.argv[1].split(',')
assert len(ids)==4 and len(set(ids))==4 and all(x.isdigit() for x in ids), 'Invalid TRAIN_GPUS'
DEVICES
export TOKENIZERS_PARALLELISM=false
AUDIT_NAME=length_audit
if [[ "$ACTION" == train ]]; then AUDIT_NAME="length_audit_train_$(date +%Y%m%d_%H%M%S_%N)"; fi
printf '正在检查数据和输入长度，日志：%s/%s.log\n' "$RUN_ROOT" "$AUDIT_NAME"
if ! "$PYTHON_BIN" -m tools.object_vlm.run audit --cache "$CACHE" --output-dir "$RUN_ROOT/$AUDIT_NAME" > "$RUN_ROOT/$AUDIT_NAME.log" 2>&1; then
  tail -n 30 "$RUN_ROOT/$AUDIT_NAME.log" >&2
  exit 1
fi
printf '数据和输入长度检查通过。\n'
run_one() {
  local phase="$1" representation="$2" run_label="$3" device_index="$4"
  CUDA_VISIBLE_DEVICES="$device_index" "$PYTHON_BIN" -u -m tools.object_vlm.run "$phase" \
    --representation "$representation" --cache "$CACHE" --device cuda:0 \
    --output-dir "$RUN_ROOT/${run_label}_$phase" > "$RUN_ROOT/${run_label}_$phase.log" 2>&1
}
run_stage() {
  local phase="$1" status=0
  run_one "$phase" text T1 "$T1_GPU" &
  local t1_pid=$!
  run_one "$phase" soft S1 "$S1_GPU" &
  local s1_pid=$!
  printf 'RUN_ROOT=%s phase=%s\nT1 GPU=%s pid=%s\nS1 GPU=%s pid=%s\n' "$RUN_ROOT" "$phase" "$T1_GPU" "$t1_pid" "$S1_GPU" "$s1_pid"
  wait "$t1_pid" || status=1
  wait "$s1_pid" || status=1
  if [[ "$status" != 0 ]]; then
    echo "Experiment failed. Inspect $RUN_ROOT/T1_$phase.log and S1_$phase.log." >&2
    return "$status"
  fi
}
run_four_rank() {
  local representation="$1" run_label="$2" phase="$3"
  local extra=()
  if [[ "$phase" == distributed_smoke ]]; then extra=(--distributed-smoke); fi
  printf 'Starting %s %s: GPUs=%s, 4 ranks\n' "$run_label" "$phase" "$TRAIN_GPUS"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 -m tools.object_vlm.run train \
    --representation "$representation" --cache "$CACHE" "${extra[@]}" \
    --output-dir "$RUN_ROOT/${run_label}_$phase" 2>&1 | tee "$RUN_ROOT/${run_label}_$phase.log"
}
if [[ "$ACTION" != train ]]; then
  run_stage smoke
  run_four_rank text T1 distributed_smoke
  run_four_rank soft S1 distributed_smoke
fi
if [[ "$ACTION" != smoke ]]; then
  # Both real smokes must succeed before either formal experiment begins.
  "$PYTHON_BIN" - "$RUN_ROOT" "$CACHE" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
from tools.object_vlm.run import base_identity
root=Path(sys.argv[1]);cache=Path(sys.argv[2]);code=Path('tools/object_vlm')
for label,representation in [('T1','text'),('S1','soft')]:
    smoke=json.loads((root/f'{label}_smoke/smoke.json').read_text())
    run=json.loads((root/f'{label}_smoke/run.json').read_text())
    assert smoke['reload_generation_equal']
    probe=root/f'{label}_distributed_smoke'
    completed=json.loads((probe/'complete.json').read_text())
    probe_run=json.loads((probe/'run.json').read_text())
    assert completed['world_size']==4 and completed['epochs']==1 and completed['steps']==3
    assert probe_run['args']['distributed_smoke']
    for name,digest in probe_run['code_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
    assert run['args']['representation']==representation
    assert run['base_identity']==base_identity(run['args']['base_model'])
    assert run['shared_manifest_sha256']==hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()
    for name,digest in run['code_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
CHECK
  for run_label in T1 S1; do
    if [[ -e "$RUN_ROOT/${run_label}_train" ]]; then
      echo "输出目录已存在：$RUN_ROOT/${run_label}_train。请先重命名保留该目录，再重启；脚本不会覆盖。" >&2
      exit 2
    fi
  done
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '启动检查全部通过；DRY_RUN=1，不启动GPU。正式训练为T1→S1，GPUs=%s，各6epoch。\n' "$TRAIN_GPUS"
    exit 0
  fi
  # Fresh BASE initialization; smoke adapters never seed the experiment.
  # Four ranks, one sample per rank. Fresh, sequential six-epoch experiments.
  run_four_rank text T1 train
  run_four_rank soft S1 train
fi
printf 'Finished. Results: %s\n' "$RUN_ROOT"
