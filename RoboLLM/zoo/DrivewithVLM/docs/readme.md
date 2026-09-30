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
## 连续轨迹数据与数值评测（C0）

生成器会保留旧版对话格式，同时在每条样本的 `planning_targets` 中写入未取整的 ego-frame `future_xy`、有效时间和 mask，以及因果历史与 ego 状态。源 PKL 路径需指向当前 V2X-Seq-SPD cooperative train/test infos；不要覆盖旧的 `v2_physics` 数据目录。

```bash
cd /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM

TRAIN_INFO=/path/to/cooperative_train_infos.pkl
TEST_INFO=/path/to/cooperative_test_infos.pkl
OUTPUT_DIR=data/Planning/v2_physics_numeric

python tools/generate_v2x_planning_v2.py \
  --train-info "$TRAIN_INFO" \
  --test-info "$TEST_INFO" \
  --output-dir "$OUTPUT_DIR" \
  --seed 42 \
  --val-scene-ratio 0.15
```

连续模型的预测 JSON 使用 token 对齐，轨迹单位为米，坐标系为当前 ego LiDAR 的 XY；每条预测包含 `token`、`future_xy`（T×2），可选 `future_times_s` 和 `future_mask`。数值评测会报告 1/2/3 秒 L2、三时域平均、3 秒 FDE、单列的 4.5 秒 FDE、逐时域有效样本数及缺失/非法预测诊断。

```bash
cd /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM

python tools/eval/continuous_trajectory_metrics.py \
  --predictions-json /path/to/predictions.json \
  --ground-truth-json data/Planning/v2_physics_numeric/test_v2x_planning_v2.json \
  --output evaluation_results/C0/continuous_trajectory_metrics.json
```

该数据生成器目前仍沿用双图像和 GT perception 文本对话。`planning_targets` 是数值监督合同，不代表样本已经转换成 ego-only 输入；C0 的 dataset/collator 还需显式屏蔽 RSU 图像和 GT 对象描述。
