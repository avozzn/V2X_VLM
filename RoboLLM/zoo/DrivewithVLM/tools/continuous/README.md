# 当前 I1 在线路侧输入与历史 M1

## 独立观测数据修复（2026-10-04，优先于下文历史数据说明）

prepare默认输出已改为`data/Planning/continuous_v3_native_gt/`，已完整导出929/157/480帧。车端框从原生`label/lidar`读取，路端框改为原生`label/virtuallidar`；路端PKL只提供frame/time/pose及因果前驱链，不提供观测框/类别/速度。保留50m、ego32/RSU16和velocity-disabled；尚未扩大候选范围、限定相机FOV或升级VLM生成接口。

新版token顺序、规划targets和RSU配对与旧v2相同，路端不存在未来消息；原生标签hash写入数据。旧v2/cache/checkpoint保持不变，不能把旧训练结果改称v3成绩。原生annotation tokens/track IDs是全量标签审计元数据，不代表与筛选后geometry逐行对应。

UniV2X单端转换已恢复train保存，完整循环后逐文件原子发布；完整兼容/Native PKL每端1521train/675val。旧车端损坏文件已备份，修复产物、测试和复现命令见[修复报告](../../docs/independent_observation_repair_20261004.md)。原生PKL的GT velocity仍为离线监督target，不能当作规划当前观测。

## 紧急接口修复（2026-10-04）

新入口：`python -m tools.continuous.online train/predict`。
`online_model.py:I1Planner`保留HF LLaVA真实图像/文本merge，在语言decoder前追加有效RSU soft embeddings和专用可训练规划embedding，再读取该位置的hidden。轨迹头输入只有语言hidden；不接收geometry、h_rsu或对象memory。对象来源zero/oracle/detector使用同构模块。当前自车对象/state/history仍为原有文本，未升级成数值soft tokens。

原始13D对象归一化→MLP→D_lm投影，另加RSU modality embedding。padding不进入有效序列；zero mode忽略所有对象，空消息与ego-only同序列。输入不接受未来答案labels/KV cache。规划embedding是独立soft参数，不是已注册到tokenizer的可生成`<wp>`文本token；本次仅连续规划，基础tokenizer未改变。

基座及可选E2 adapter冻结；训练agent encoder、规划embedding、RSU embedding和轨迹head。语言forward保留梯度以训练新增输入。此入口**尚未开放语言LoRA联合更新**，也未实现age质量token、时间补偿、I2/I3、状态encoder或VAE。沿用经审计的velocity-disabled数据版本，消息时间不得在未来。

### 训练示例（未执行）

先核对GPU/内存，使用独立输出目录；batch=1起步，旧hidden cache不参与新训练。

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.continuous.online train \
  --train-samples data/Planning/continuous_v2_ego_gt/train.json \
  --val-samples data/Planning/continuous_v2_ego_gt/val.json \
  --rsu-mode oracle --device cuda:0 --batch-size 1 \
  --output work_dirs/I1_oracle_online_new
```

需要冻结E2驾驶权重时追加`--adapter checkpoints/meta_v2_physics_seed42_4gpu/checkpoint-932`。
F0改为`--rsu-mode zero`；Detector需真实预测来源cache，不能只改source标签。

### 重载预测示例（未执行）

best.json记录实际checkpoint子目录；下面checkpoint-N需替换为其值。dtype与训练一致。

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.continuous.online predict \
  --checkpoint work_dirs/I1_oracle_online_new/checkpoint-N \
  --samples data/Planning/continuous_v2_ego_gt/val.json \
  --device cuda:0 --output work_dirs/I1_oracle_online_val_predictions.json
```

保存i1.pt（所有新增模块）和i1.json（基座/adapter身份、interface、配置）；基座文件复用，重载核验身份。预测JSON兼容统一连续轨迹评测器。

旧`ContinuousPlanner`和缓存train入口已拒绝oracle/detector旁路；旧F0 zero模型仍可重载核验历史结果。新I1不能加载旧planner.pt或用旧h_ego cache伪装路侧已进入VLM。下文完整保留旧文档，仅用于历史复现，旧F1/F2命令已失效。

验证：15项CPU测试通过，含真实tiny HF LLaVA（Llama/CLIP及本地同族Qwen2/SigLIP）forward、对象与图像敏感性、masked padding、empty/zero一致、loss反传、冻结基座无梯度、保存重载与旧旁路拒绝。不是7B真实数据smoke，也不是正式规划性能结论。

---

# 同构 F0/F1 连续规划实现

此入口独立于旧文本 HF Trainer。复用已有
`projects/Robodrivevlm/model/continuous_planner.py` 的 waypoint head，
用冻结 LLaVA 的缓存表示训练同构 F0/F1/F2，不改旧训练的行为。

## 实现范围

- `prepare.py`：以旧 V2 physics 的 token 列表锁定原 929/157/480 split，
  从 PKL 重建未四舍五入的 numeric targets，从独立路侧 GT 读取对象。
- `model.py`：从实际图像合并后的 decoder mask 定位最后有效上下文 token；
  冻结的 final-normalized hidden → 同构 D+256 head。
- `ego_perception.py`：读取独立车端原生 LiDAR GT，统一 GT/Detector 文本合同。
- `data.py`：读取车端图像、车端对象文本、因果自车状态/历史和规划提示；不读 assistant 答案或路侧对象。
- `cache_features.py`：真实预训练 LLaVA 仅观测 forward，缓存 train/val。
- `train.py`：独立初始化的 zero/oracle/detector 实验，唯一累计 waypoint
  Smooth-L1；按 validation L2@1/2/3 s 平均选 best；保存 head/encoder/scales/config。
- `predict.py`：重新加载 planner 并输出现有数值 evaluator 接受的 JSON。

`ContinuousPlanner` 三种 mode 的 state_dict 结构、参数数目和初始化相同。
zero mode 保留 AgentEncoder，强制 h_rsu=0；没有有效 agent 梯度是预期行为。
缓存 backbone eval/no_grad，只有新规划分支训练。train 默认 CPU，可以在
GPU 忙时训练缓存特征；目前为单进程，不应直接用 torchrun 启动 train CLI。
模型 loss helper 支持 DDP 全局有效点归一化，但 CLI 尚未实现 DDP sampler。

## 当前车端 GT 验证协议

车端输入是图像 + vehicle-side 原生 LiDAR GT 对象文本 + 自车尺寸/速度/加速度/历史
+ 固定规划提示。Ego-only 指没有路侧输入，车端对象文本在 F0/F1/F2 中完全相同。
这是 **Oracle ego perception** 验证，不能作为实际 detector 性能报告。

车端 GT 仅从 `vehicle-side/data_info.json` 指定的 `label/lidar` 文件读取，严格
匹配当前 ego LiDAR timestamp；不读取 cooperative 标签或 GT future object tracks。
车端默认 ROI=50m、最近最多32对象（`--ego-max-objects`），直接使用原生 xyz/lwh/
物理 heading。此为车端 LiDAR 标注覆盖范围，并未限定为前向相机可见对象；
不能称为 image-only planner。对象速度和检测置信度没有可靠观测时文本为 unknown。

`ego_perception` 独立于路侧 `agents`，带 source/frame/timestamp 和 label 路径。
未来冻结车端 detector 后，将预测转成相同 objects 字段，source 改为
`vehicle_detector`；GT 和 detector 使用相同 formatter，不需要先在 VLM 内训练
检测头。还需要单独实现真实预测导出器，不能只改 GT 的 source 标签。

prompt version 已升级；continuous_v1 及其旧 feature cache 不再用于当前训练。
旧数据留作历史 smoke 记录。当前默认新输出为 continuous_v2_ego_gt。

## 路侧数据合同和第一版限制

对象为 `[xyz,lwh,sin_heading,cos_heading,vx,vy,class_one_hot]`，K=16、
ego 平面半径 ROI=50m。Undo UniV2X stored yaw=-annotation_yaw-pi/2；
尺寸从 wlh 改为 lwh。heading 为物理长度轴，不是直接传 detector 编码角。

第一版明确采用 **velocity-disabled geometry setting**：所有对象 vx/vy=0，
保留 13D shape，但不将未知速度装成可信观测。源 GT velocity 可能通过未来
帧计算，因此不能直接读取。后续接入经审计的因果 track velocity 前，Oracle
和 Detector 都必须使用此策略，报告中不能称为完整速度感知模型。

原生车路配对的路側时间有时晚于车端。exporter 选择当前时刻已经可用的
配对/前驱路侧消息，使用世界坐标将选定 source frame 变换到配对 infra frame，
再用已审计标定转换到 ego；没有历史时为空对象。当前不做运动补偿，消息年龄
只作审计元数据，不能称为同步当前时刻的完美感知上界。manifest 报告 age 范围。
本次构建的旧数据最大消息年龄约 2.2s；分析 Oracle 无收益前应检查此分布。

旧 V2 数据保持不动，新增在 `data/Planning/continuous_v2_ego_gt/`；导出器拒绝覆盖。
F2 的代码输入合同已预留，但本次不包含 D2 native prediction→13D exporter，
不能把 GT 缓存改个 source 名称当成 detector 实验。

## 运行顺序

数据已导出到 `data/Planning/continuous_v2_ego_gt/`，无需再次 prepare。
使用本地 robollm Python（PyTorch 2.3.0、transformers 4.45.2）。当前 SigLIP/full
checkpoint 保留 legacy image merge，在 decoder 入口读取真实 mask；升级
transformers 时需重新验证该接口。

### 1. 进入项目目录

```bash
cd /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM
PY=/home/zzn/anaconda3/envs/robollm/bin/python
```

如需重新导出，指定一个新的版本目录：

```bash
"$PY" -m tools.continuous.prepare --output-dir data/Planning/continuous_v2_ego_gt_new
```

### 2. 缓存完整 train/val 特征

```bash
nvidia-smi
```

以下以物理 GPU 0 为例；将 `CUDA_VISIBLE_DEVICES=0` 改成空闲卡编号。
`--device cuda:0` 表示进程可见的第一张卡。两个命令依次运行。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.continuous.cache_features \
  --samples data/Planning/continuous_v2_ego_gt/train.json \
  --output data/Planning/continuous_v2_ego_gt/train_features.pt \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch-size 1
```

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m tools.continuous.cache_features \
  --samples data/Planning/continuous_v2_ego_gt/val.json \
  --output data/Planning/continuous_v2_ego_gt/val_features.pt \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch-size 1
```

缓存包含车端图片、车端GT对象文本、自车状态/历史和规划提示。F0/F1共用这些
特征，只需提取一次。已有 `train2_smoke_features.pt`、`val2_smoke_features.pt`
只有两条样本，不能用于以下完整训练。缓存文件已存在时脚本拒绝覆盖。

### 3. 独立训练 F0

只训练小型规划分支，可以使用 CPU：

```bash
"$PY" -m tools.continuous.train \
  --train-samples data/Planning/continuous_v2_ego_gt/train.json \
  --val-samples data/Planning/continuous_v2_ego_gt/val.json \
  --train-features data/Planning/continuous_v2_ego_gt/train_features.pt \
  --val-features data/Planning/continuous_v2_ego_gt/val_features.pt \
  --rsu-mode zero \
  --output-dir work_dirs/continuous_F0_ego_gt_seed42 \
  --epochs 100 \
  --batch-size 32 \
  --head-dim 512 \
  --lr 0.001 \
  --seed 42 \
  --device cpu
```

结果目录：

- `history.jsonl`：每轮训练 loss、validation L2、逐时域误差和预测方差。
- `run.json`：运行参数、来源、恒速/直线/训练均值基线。
- `best_metrics.json`：最佳 validation 指标。
- `val_predictions.json`：最佳模型的9点轨迹预测。
- `best/`、`last/`：规划分支参数、配置及特征来源 metadata。

先检查 F0 是否可学习、是否优于恒速/直线基线、是否坍缩到均值，并检查9点
轨迹可视化。F0 未通过正式验收，不训练 F1。重复实验使用新输出目录。

### 4. F0 验收后独立训练 F1

车端GT、特征、网络、初始化 seed 和训练参数保持一致，只切换路侧来源。

```bash
"$PY" -m tools.continuous.train \
  --train-samples data/Planning/continuous_v2_ego_gt/train.json \
  --val-samples data/Planning/continuous_v2_ego_gt/val.json \
  --train-features data/Planning/continuous_v2_ego_gt/train_features.pt \
  --val-features data/Planning/continuous_v2_ego_gt/val_features.pt \
  --rsu-mode oracle \
  --output-dir work_dirs/continuous_F1_ego_gt_seed42 \
  --epochs 100 \
  --batch-size 32 \
  --head-dim 512 \
  --lr 0.001 \
  --seed 42 \
  --device cpu
```

### 5. 重新加载预测并使用统一 evaluator

```bash
"$PY" -m tools.continuous.predict \
  --checkpoint work_dirs/continuous_F0_ego_gt_seed42/best \
  --samples data/Planning/continuous_v2_ego_gt/val.json \
  --features data/Planning/continuous_v2_ego_gt/val_features.pt \
  --output work_dirs/continuous_F0_ego_gt_seed42/reloaded_predictions.json

"$PY" -m tools.eval.continuous_trajectory_metrics \
  --predictions-json work_dirs/continuous_F0_ego_gt_seed42/reloaded_predictions.json \
  --ground-truth-json data/Planning/continuous_v2_ego_gt/val.json \
  --output work_dirs/continuous_F0_ego_gt_seed42/unified_metrics.json
```

checkpoint 包含规划参数和特征生成 metadata，不复制冻结的 7B 权重；重新
生成特征时需使用相同模型路径、权重版本、processor、prompt/readout。
缓存校验 model config、preprocessor、权重文件 size/mtime、输入图像路径、
token/timestamp 和 prompt；权重身份是文件统计，不是全内容哈希，图片路径
相同而图片内容被替换时应主动重建缓存。

当前保存支持预测恢复，不支持从中断 epoch 恢复 optimizer/scheduler。
新实验拒绝覆盖已有输出目录。

## 验证

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m pytest tests/test_continuous_planner.py tests/test_rsu_continuous_pipeline.py tests/test_planning_metadata_v2.py -q
```

包含：F0/F1 同构初始化、padding/空集合/置换不变、两个样本均反传、
mask 无效 NaN 不进入 loss、保存加载、GT 回答隔离、实际 tiny LLaVA 合并和
左右 padding readout、坐标/尺寸/yaw 数值验证。tiny 模型随机权重测试
不能替代真实 7B 特征的 F0 学习能力验证。
