## 路侧检测头训练

```bash
cd /home/zzn/V2X_VLM/UniV2X

INF_WORK_DIR=/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/work_dirs/e5a_d1_d2_q3_full_8gpu_e30/infrastructure

test -f "$INF_WORK_DIR/latest.pth" || {
  echo "未找到 D2 checkpoint：$INF_WORK_DIR/latest.pth"
  exit 1
}

PYTHONPATH=/home/zzn/V2X_VLM/UniV2X \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_LEVEL=NVL \
/home/zzn/anaconda3/envs/univ2x/bin/python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  tools/train.py \
  /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/projects/configs/Robodrivevlm/univ2x_e5a_infrastructure_det.py \
  --launcher pytorch \
  --deterministic \
  --no-validate \
  --work-dir "$INF_WORK_DIR" \
  --resume-from "$INF_WORK_DIR/latest.pth" \
  --cfg-options \
    data.train.queue_length=3 \
    model_ego_agent.queue_length=3 \
    data.workers_per_gpu=4 \
    total_epochs=30 \
    runner.max_epochs=30 \
    checkpoint_config.interval=1 \
    log_config.interval=10
```

## 车侧检测头训练

```bash
cd /home/zzn/V2X_VLM/UniV2X

PYTHONPATH=/home/zzn/V2X_VLM/UniV2X \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_LEVEL=NVL \
/home/zzn/anaconda3/envs/univ2x/bin/python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29504 \
  tools/train.py \
  /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/projects/configs/Robodrivevlm/univ2x_e5a_vehicle_det.py \
  --launcher pytorch \
  --deterministic \
  --no-validate \
  --work-dir /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/work_dirs/e5a_d1_d2_q3_full_8gpu_e30/vehicle \
  --resume-from /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/work_dirs/e5a_d1_d2_q3_full_8gpu_e30/vehicle/latest.pth \
  --cfg-options \
    data.train.queue_length=3 \
    model_ego_agent.queue_length=3 \
    data.workers_per_gpu=4 \
    total_epochs=30 \
    runner.max_epochs=30 \
    checkpoint_config.interval=1 \
    log_config.interval=10
```
