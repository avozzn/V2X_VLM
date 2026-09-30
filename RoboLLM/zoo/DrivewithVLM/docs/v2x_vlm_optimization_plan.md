# V2X-VLM 下一阶段优化计划

## 实施进展（持续更新）

> 最后更新：2026-09-30。每次完成本项目的实现、训练、评测或分析任务后，都应在本节更新状态、产物路径和下一步；“已实现”不等于“已验证”。本次同步：E2/V2 physics 的 6-checkpoint validation sweep 已完成并选出 checkpoint-932；E2 test 最终产物仍待生成。E5-A 的 D1/D2 已完成统一 `queue_length=3` GPU smoke；它证明三帧单端训练链路可运行，但尚未产生 detection/track 指标或全量 checkpoint。

| 阶段/实验 | 状态 | 已有产物或结论 | 下一步 |
|---|---|---|---|
| E0：旧版可靠 baseline | 已完成 | 已在 `test_dt_resampled_4.json` 上完成 6 个 checkpoint 的评测。按全 9 步平均 L2 选出的 baseline 是 `V2X_dt_9step/checkpoint-1552`：coverage 100%（675/675）、mean L2 1.4264 m、4.5 s L2 3.4648 m、collision-any 7.11%、非法轨迹率 0、旧 Meta Action accuracy 43.26%。结果：`evaluation_results/E0_fixed/gpu3/checkpoint-1552/diagnostics.json`。 | 将 E0 结果汇总为单一报告，并保持测试集只作最终报告用途。 |
| E1：因果历史 | 已实现，未形成独立对照报告 | `train_v2x_causal_history.json`、`test_v2x_causal_history.json` 已生成；历史 mask 和连续性统计已存在。 | 在固定、可比较的 split 上训练/评测 E1，并与 E0 分表报告。 |
| E2：动作拆分 | validation 已验证；test 待最终报告 | V2 标签、prompt、解析器和单测已使用 `Lateral Action` + `Longitudinal Action`；6 个 checkpoint 已在 157 样本 validation set 上 sweep，按预注册规则选择 `checkpoint-932`。其 mean L2 为 1.1442 m、末点 L2 为 3.2208 m、横向/纵向文本 macro-F1 为 49.57%/63.77%。 | 仅用 checkpoint-932 完成 480 样本 test，随后与 E0 按相同数据协议做正式对照。 |
| E2：类别均衡 | 未开始 | 审计已显示稀有类（如 `STOP`、右变道），但尚未发现 planning 的 weighted sampler/过采样配置。 | 实现类别均衡，作为单变量实验重新训练。 |
| 新版 V2 physics 训练 | validation 完成；test 待最终报告 | `checkpoints/meta_v2_physics_seed42_4gpu/` 的 6 个 checkpoint 已完成 validation sweep，汇总见 `evaluation_results/meta_v2_physics_seed42_4gpu_val/sweep_summary.json`，选择结果见同目录 `best_checkpoint_selection.json`。按 mean L2 选择 checkpoint-932，不应以 HuggingFace trainer state 中的 checkpoint-466 替代。 | 仅用 checkpoint-932 跑 `test_v2x_planning_v2.json`，保存完整 diagnostics；test 与旧 E0 测试集不同，必须分表或重跑统一协议才可声称优于 E0。 |
| E3：route/lane | 实施前检查完成，尚未实现 | UniV2X 已有可行驶区域、车道和停止线的地图/分割基础能力；但 DrivewithVLM 的 planning JSON（仅含 token、scene、image、planning labels、conversations）及 prompt 尚未接入 route、导航命令或 lane 特征。V2/E2 的选择性 Git checkpoint 已建立，未包含数据、权重、日志或无关改动。 | 确认可导出的地图/route 来源和车辆坐标系；随后实现 E3 数据生成与结构化 prompt。 |
| E4：连续轨迹回归头 | 未开始 | 当前 V2 仍由 LLM 文本输出 `Trajectory`；没有新的 `[B, 9, 2]` trajectory head 或对应 loss。 | 在 E3 通过验收后实现回归 head 和运动学/碰撞 loss。 |
| E5：3D 检测多任务融合 | A-D0、D1 完成；D2 单帧 smoke 通过；D3 不作 Proposed 主线 | 新增只读审计 `tools/eval/audit_v2x_detection_d0.py`，报告：`evaluation_results/E5_d0_univ2x/train_coordinate_audit.json`；新增 `univ2x_e5a_vehicle_det.py`、`univ2x_e5a_infrastructure_det.py` 与显式 `SPDE2EDetectionDataset`。车端/路端均成功取到 1,521 帧数据及 5 帧时序检测样本；检测专用路径不再生成 map/occupancy/planning 标签。D1 的 `PYTHONPATH`、epoch 断言、debug index-0/5-frame queue 死循环均已修正；2026-09-28 23:05 的 GPU 4 单 iteration 已输出完整 `track.frame_* loss`、`loss=127.5410`、`grad_norm=977.8541` 并保存 checkpoint，证明 vehicle-only 端的数据、前向、loss 与 backward 链路可运行。D2 的原始 5 帧、1600×900、200×200 BEV、900 query、batch=1 配置在 GPU 4（进程内显示为 GPU 0）峰值需额外 230 MiB 时 OOM；当时 PyTorch 已分配 45.06/47.43 GiB，并非物理 GPU 0 被占用。用户已报告 D2 的 `queue_length=1` 单帧 smoke 成功；单帧仅是显存/数据链路验证，不是最终时序设定。下一步优先验证 `queue_length=3`（官方 cooperative config 的时序长度），再决定 D1/D2 统一用 3、2 或 1 帧；不能依赖增加 GPU 数量解决单卡 OOM。D3 的标定并不缺失：cooperative pkl 的 `other_agent_info_dict['model_other_agent_inf']` 已含 `VehLidar2InfLidar_rotation/translation`（由原始车端/路端 global calibration 和 `system_error_offset` 计算），D0 已对 1,521 帧验证其往返误差最大 0.000213 m。当前检测专用 dataset 对 other agent 可取到该字段，但 ego 分支仍未从嵌套字典导出，且 cooperative pkl 没有路端 GT；现有 `MultiAgent.forward_train` 会以 `return_loss=True` 调用冻结路端模型，导致路端分支使用车端 GT 形成 tracking query。这不等价于 D2 checkpoint 的无标签路端 query 推理，不能作为有效 D3。 | 先以 3 帧对 D1/D2 做 smoke 与 2-iteration 吞吐测试；若 3 帧 OOM，再顺序尝试 2 帧和 1 帧，并且三种实验必须使用同一时序长度。D1/D2 只需完成最小 detection/track 评测及 RSU 输出到 ego geometry-token 的导出；Proposed 主线先进行 oracle token → VLM/continuous planner。将 D3 direct query/BEV fusion 改作 UniMM 类的可选诊断 baseline，只有在需要比较 detector query semantic feature 时才实现 inference-only 路端 query path。当前不生成包含假 D3 的三段全量脚本。 |
| 后续模型与论文实验设计 | 已完成方案，不代表实现 | 新增 `docs/v2x_vlm_model_optimization_and_experiment_plan.md`：确定“ego-frame、time/uncertainty-aware dual Agent Token + VLM semantic latent + continuous displacement planner”路线；主基线选 OmniV2X、UniMM-V2X 与 V2X-VLM-style；给出比较、消融、VLM/数据集、鲁棒性和可选闭环表格。 | 先完成 E2 test 与统一 1/2/3 s evaluator，再按该文档执行 C0/C4/C6。 |

### D1/D2 全量启动失败诊断与短程修复验证（2026-09-30，优先于下文历史状态）

9 月 28 日启动的四卡 q=3 全量任务并未完成：D1 在 9 月 29 日 00:08:13 因 NCCL BROADCAST 等待 1800 秒超时而 SIGABRT；无训练指标或 checkpoint，串行脚本未进入 D2。9 月 30 日检查无残留训练进程。独立 4 MiB 张量通信测试在默认 P2P 设置下复现阻塞；`NCCL_P2P_DISABLE=1` 和 `NCCL_P2P_LEVEL=NVL` 均通过 broadcast/all_reduce。日志与拓扑将问题定位至当前环境的跨 PCIe host bridge P2P 路径，尚未判定 ACS/IOMMU/驱动/NCCL 中的底层根因。

启动脚本现默认 `NCCL_P2P_LEVEL=NVL`（支持环境覆盖），保留 4↔7、5↔6 的 NVLink P2P，其余连接使用共享内存。D1/D2 均已在此设置下完成四卡、q=3、workers=4、8 个 debug 样本的两步训练，并以 exit 0 保存 `/tmp/e5_d1_nvl_ddp_probe/epoch_1.pth` 和 `/tmp/e5_d2_nvl_ddp_probe/epoch_1.pth`；这是短程链路验证，不能记作 40 epoch 完成或检测精度验证。完整证据：[诊断报告](../evaluation_results/E5_ddp_diagnosis_20260930/report.md)。后续 D2 NaN 已定位并修复，见下一节；本次未启动全量训练，完整任务由用户执行。

### D2 NaN 根因修复与最终短程验收（2026-09-30）

autograd anomaly 将 NaN 定位到 `UniV2XTrack.velo_update` 中 `inverse_sigmoid` 的 `DivBackward0`：跨帧几何在 autocast 下接近坐标边界时出现半精度反向不稳定，初始 loss scale 从 512 降为 32 或 1 均不能单独修复。已为该函数加 `force_fp32`，同时将 D2 的固定 loss_scale=512 改为动态缩放（init_scale=32、growth_interval=2000），处理修复几何后剩余的缩放溢出。D1 原本使用 FP32，保持原精度。

最终验证：边界/越界坐标的 AMP 梯度回归通过；原 D2 失败样本两步复测四 rank 均无非有限梯度、无跳步；D1/D2 各 80 个 debug 样本、四卡 q=3、workers=4、每 rank 20 步全部梯度/参数有限，每步均更新并保存 checkpoint。最终 checkpoint 的模型与优化器浮点状态也全部有限。证据：[数值诊断报告](../evaluation_results/E5_numeric_diagnosis_20260930/report.md)、[逐 rank 汇总](../evaluation_results/E5_numeric_diagnosis_20260930/summary.json)。这是短程验证，尚非全量训练或检测精度报告。

启动脚本 `tools/train/run_e5a_d1_d2_full_q3.sh` 已默认使用 NVLink P2P，输出至 `work_dirs/e5a_d1_d2_q3_full_ddp4_stable/`，保留原失败产物。默认 GPU 4–7、D1→D2 串行、每端 40 epochs、q=3；`RESUME=1` 从此新目录恢复。诊断任务全部结束；用户自行启动全量任务，完成后另做 detection/track validation（脚本仍为 `--no-validate`）。

### 当前阻塞点与最近下一步

当前最明确的“改好了但还未形成最终报告”的功能是 **V2 physics 的 test 评测**：validation selection 已完成，checkpoint-932 已唯一确定；现在只需在冻结 checkpoint-932 的前提下产出 test diagnostics。test 结果出现前，不能以 validation 数字直接宣称对 E0 的最终提升。

注意：E0 使用 675 个旧测试样本；V2 physics 的 test split 为 480 个完整 4.5 s 样本。这两套结果不能直接混排，应固定同一评测集或分表报告。

#### E5 三帧 smoke 后的决策门（2026-09-28）

用户已报告 D1 与 D2 的统一 `queue_length=3` GPU smoke 成功；这覆盖了三帧数据打包、前向、检测/跟踪 loss 与 backward，但**不是**检测精度验证。后续不直接启动 D1/D2/D3 三段全量训练，原因是 Proposed 的核心问题是“RSU geometry token 是否能改善 continuous planner”，而非先追求单端 detector 指标；若 oracle token 对 ego-only planner 无收益，完整训练 D1/D2 或替换为 UniMM 都不能回答该问题。

执行顺序固定为：

1. 完成冻结 `checkpoint-932` 的 E2 test 与统一 1/2/3 s evaluator；
2. 完成 E3 可部署 lane/route 输入、E4 continuous ego-only planner（C0）；
3. 从 infrastructure-side 的成对 GT 构造并经 `veh2inf_rt` 逆变换到 ego frame 的 oracle RSU geometry tokens，训练 C6；
4. **并行的感知线：** 立即为 D1/D2 补 detection-only validation/evaluator，先训练短 pilot（建议 1 epoch 或预先固定的小 iter 预算）并保存可评测 checkpoint，报告 3D detection/track 指标、显存和吞吐；smoke/loss 不能证明 head 有效。若指标和训练曲线正常、且决定 D1/D2 是最终 detector-token 前端，再投入完整单端训练。UniMM 另按官方 stage-1/2 作为 C3 baseline，不能仅替换一个 head。
5. **并行的规划线：** 构造 oracle RSU geometry tokens 并训练 C6；其作用是判断 detector-token 注入 planner 是否值得，而不是替代 D1/D2 的检测评测。C6 的收益决定 D1/D2 full checkpoint 是否还值得继续接入 Proposed，不能倒过来把 smoke 当成最终 detector 结论。

因此，D1/D2 的完整训练被**分为 pilot → detection/track 评测 → full training**，而非取消；D3 direct query/BEV fusion 保持为可选 UniMM 类诊断，不作为 C6/C7 主线的前置条件。

#### E5-A D1/D2 全量训练脚本（2026-09-28）

用户在 D1/D2 的统一 3 帧 smoke 成功后决定直接启动完整单端训练。已新增 `tools/train/run_e5a_d1_d2_full_q3.sh`（仅创建，未由本任务执行）：默认先以物理 GPU 4--7 的 4-rank DDP 训练 D1 vehicle，完成后再以同样 4 卡训练 D2 infrastructure；二者均为 `queue_length=3`、每 GPU `batch=1`、40 epochs、每 epoch 保存 checkpoint，日志分别写入 `work_dirs/e5a_d1_d2_q3_full_ddp4/logs/`。采用官方 `torch.distributed.launch + --launcher pytorch` 入口；4 卡提升吞吐，但不合并显存，故不能恢复 D2 的 5 帧设定。`RESUME=1` 时会自动从当前 stage 的 `latest.pth` 恢复。该脚本刻意不包含 D3；D3 的 inference-only 路端 query 语义尚未实现，不能作为有效的夜跑任务。

### 配置与数据集单一事实来源（2026-09-28 整理）

所有下列 config 的 `data_root` 都是 `data/V2X-Seq-SPD-New/cooperative/`，即指向 `UniV2X/datasets/V2X-Seq-SPD-New` 的软链接；**当前训练并未使用旧的 `data/V2X-Seq-SPD`**。

| Config | 所属阶段 | 状态与用途 |
|---|---|---|
| `MMdrive_v2x.py` | E0 / 旧版基座 | 旧 Meta Action、causal-history JSON 的基座；E0 的 6-checkpoint 评测使用此 config，并以命令行覆盖为 `test_dt_resampled_4.json`。仅用于复现旧 baseline。 |
| `MMdrive_v2x_det.py` | 历史检测文本实验 | 使用 `train_v2x_dt_VM_re4.json` / `test_v2x_resampled_4.json` 的旧检测提示变体，时间早于 V2 metadata 工作；不作为 E1--E5 的入口。 |
| `MMdrive_v2x_meta_v2.py` | E2 中间基座 | 从 `MMdrive_v2x.py` 继承，切换到 scene-disjoint 的 `data/Planning/v2/{train,val,test}_v2x_planning_v2.json`，保留旧输出合同。未发现其独立训练 checkpoint。 |
| `MMdrive_v2x_meta_v2_physics.py` | 当前 E2 | 从 `meta_v2` 继承，使用 `data/Planning/v2_physics/{train,val,test}_v2x_planning_v2.json`，输出为横向动作 + 纵向动作 + 9 点轨迹；`meta_v2_physics_seed42_4gpu` 是其已训练 checkpoint。E3 应从此 config 派生新 config。 |

`V2X-Seq-SPD-New` 中存在 `commands_test.pkl`，含 0=TURN_RIGHT、1=TURN_LEFT、2=KEEP_FORWARD 三类高层命令；但当前 DrivewithVLM planning 数据没有读取它，且当前 `spd_trajectory_api.py` 也会由未来轨迹终点推导同类 command。因此在 E3 将其作为模型输入前，必须确认训练/验证/测试三套命令均为部署时可获得的导航标签，不能将由未来 GT 推导的 command 当作 route 输入。

### E3 前置 GT 审计（2026-09-28，只读）

| 信息 | 是否存在 | 来源与当前坐标系 | 当前 VLM 是否实际使用 |
|---|---|---|---|
| 3D 目标检测 GT | 是 | `spd_infos_temporal_{train,val}_sdc.pkl` 的 `instances[*].bbox_3d`、类别、速度和有效标记。box 为 `[x,y,z,w,l,h,yaw]`，在当前车端 LiDAR / 自车局部坐标系；prompt 中的 `Front-Left (x,y)` 同样使用该坐标，X 向前、Y 向左。 | 部分使用：`Processed_sensor` 将 GT 框转成检测文本；当前 VLM 并没有接入检测 head/query。 |
| 车道、可行驶区域、停止线 GT | 是，但不在每条 planning JSON 中 | 静态来源为 `V2X-Seq-SPD-New/maps/yizhuang*.json`。`SPDE2EDataset` 通过当前 `ego2global_translation/rotation` 取朝向对齐的 102.4 m × 102.4 m 地图 patch，再 rasterize 到 200×200 的 ego-BEV mask；该 GT 由 `gt_lane_labels`、`gt_lane_bboxes`、`gt_lane_masks` 传给 panoptic segmentation head。 | 否：当前 DrivewithVLM 使用 `SpdVehicleE2EVLMDataset` 与 `Processed_sensor`，其 pipeline 未产生/收集 `gt_lane_*`。 |
| route / 导航命令 | 仅发现测试命令文件 | `commands_test.pkl` 仅有 0/1/2 三类命令；训练/验证 info 没有独立 command 字段，现有 `SPDTraj` 会根据**未来 GT 终点**推导 command。 | 否，且不能把未来 GT 推导值作为部署时的导航输入。 |

因此 E3 的正确起点不是重新生成静态地图文本，而是先把已有 UniV2X 的 `gt_lane_*` 生成路径和预测 `seg_head` 输出接入 DrivewithVLM；训练时用 GT 监督，推理时只能给规划模块使用预测 lane/drivable features。route command 必须另行确认其训练/验证来源后才可接入。

### E5：3D 检测与车路协同架构评估（2026-09-28，只读）

目标检测的**统一输出坐标系应固定为车端 LiDAR / 自车局部坐标系**，与当前规划轨迹及 `gt_bboxes_3d` 保持一致。路端不是第二个“相机视角”这么简单：其相机、虚拟 LiDAR 和 BEV 均有自己的局部坐标原点，必须在检测 query 级别或 BEV 级别完成刚体坐标变换后再融合。

| 代码库 | 检测 head 与 VLM 的关系 | 对车端/路端坐标差异的处理 | 结论 |
|---|---|---|---|
| `zoo/OmniDrive-main` | `Petr3D` 从 EVA 图像特征调用 `StreamPETRHead`，以 `gt_bboxes_3d/gt_labels_3d` 计算检测 loss，并将 `det_query` 与地图 query 拼接后送入 LLM。 | V2X dataset 会追加路端图像，但该 `Petr3D` 配置和模型没有使用 `veh2inf_rt`、没有双 agent 模型，也没有跨 agent query fusion；路端相机投影未见明确变换到车端检测坐标系的完整链路。 | 可借鉴“检测 head + detection queries + LLM”的接口；**不可直接作为可靠的车路协同坐标融合方案**。 |
| `UniV2X` | 车端和路端各自运行 `UniV2X/BEVFormerTrackHead`；路端产出 track instances/query，车端检测/跟踪 head 继续输出车端结果。 | dataset 从路端 info 读取 `VehLidar2InfLidar_rotation/translation` 并构造 `veh2inf_rt`。`AgentQueryFusion` 先按各自 `pc_range` 反归一化路端 reference points，使用 `inv(veh2inf_rt.T)` 将其变换到车端，再按车端 range 归一化、Hungarian 匹配、旋转矩阵条件化并融合/补充 query。 | 这是当前项目中已实现且与问题匹配的协同方案，应作为 E5 的坐标与融合参考。 |
| 当前 `DrivewithVLM` | 训练/推理由 HuggingFace LLaVA 脚本直接处理图片和文本；`Processed_sensor` 仅把 GT 目标转为文字。 | 虽然数据集类会读取车路图像及标定，但 LLaVA 视觉 token 没有使用 BEV、`veh2inf_rt` 或 detection query；没有可直接挂接 `StreamPETRHead` 的共享特征接口。 | 必须新建明确的多任务模型/训练入口，不能把 head 塞进现有 physics config 就期望生效。 |

UniV2X 的实际协同范围也要如实限定：它采用**路端独立感知、车端融合**，路端模型在训练中由 `torch.no_grad()` 执行（冻结），融合对象主要是检测/跟踪 query；不是原始图像特征直接融合，也不是将两张图按 camera batch 拼接后自动解决坐标问题。配置中车端 BEV range 为 `[-51.2, -51.2, -5, 51.2, 51.2, 3]`，路端 range 为 `[0, -51.2, -5, 102.4, 51.2, 3]`，因此两端输出在融合前绝不能直接比较其归一化 reference point。

两者没有“谁整体更先进”的单一结论。OmniDrive 的 `StreamPETRHead` 是以图像 3D position embedding、temporal memory 和 DETR query 为核心的单模型检测头，优势是它直接返回 `det_query` 并已接到 LLM；UniV2X 的 `BEVFormerTrackHead` 是带 BEV encoder、时序跟踪和未来轨迹状态的检测/跟踪头，优势是车端与路端各自预测后能经标定在 query 层协同。对本项目应采用“**UniV2X 解决协同坐标与融合，OmniDrive 借鉴 `det_query → VLM/trajectory` 接口**”，而不是二选一整体迁移。

**E5 选型决策：** 目前不能、也不应仅凭代码宣称某个单视角检测头精度更高，需在同一 SPD split、类别集合、GT、图像分辨率、训练预算和车端坐标评测协议下对照。第一版实现固定使用 **UniV2X 的同构车端 + 路端 `BEVFormerTrackHead`**：其 query 表示、reference point 和 `AgentQueryFusion` 已经相互匹配，路端仅使用不同的 `pc_range`。不要将“车端 OmniDrive、路端 UniV2X”直接混搭；二者 query 的特征语义、box 编码、reference point 和时序状态均不保证兼容，会把检测头差异和融合适配错误混为一谈。

第二候选为“经 D0 标定闭环修复后的 OmniDrive `StreamPETRHead` 双端实现”，而不是直接运行现有 `mask_eva_lane_det_vlm_v2x.py`：该文件可作为检测-query-to-LLM 参考，但现有 V2X 路端相机路径未见完整的 `veh2inf_rt` 协同融合使用。对照实验应按以下顺序进行：

1. **A / 必做：** UniV2X vehicle-only、infrastructure-only、cooperative 三组；先证明每一端和协同融合可用。
2. **B / 条件做：** 在修正跨端投影、使用相同 GT/range/evaluator 后，训练 OmniDrive 双端检测分支；先做单端，再实现等价 query 融合。
3. 选型主指标为车端坐标系下的 3D detection mAP/误差、遮挡和远距离目标子集指标、推理显存/时延；cooperative 相对 vehicle-only 的增益是硬约束。只有 B 在同等预算下优于 A，才替换检测主体；否则保留 A，并单独移植 OmniDrive 的 `det_query → VLM/trajectory` 接口。

#### E5-A / D0 结果：官方检测 GT 与协同标定（2026-09-28）

已运行：

```text
python tools/eval/audit_v2x_detection_d0.py \
  --ann-file UniV2X/data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_train.pkl \
  --output evaluation_results/E5_d0_univ2x/train_coordinate_audit.json
```

审计对象是 UniV2X 检测配置实际读取的 `spd_infos_temporal_train.pkl`（`infos/metadata` 格式），不是 DrivewithVLM 使用的 `*_sdc.pkl`（`data_list/metainfo` 格式）。统计结果：1,521/1,521 帧具备 `VehLidar2InfLidar` 标定；有效 GT 35,894 个；车端 range 内为 19,332 个；将 GT 由车端 LiDAR 坐标变换后落入路端 range 的为 26,301 个；若错误地以未变换的车端坐标直接按路端 range 筛选则为 22,639 个。车辆→路端→车辆最大闭环误差为 **0.000213 m**，旋转矩阵正交/行列式误差均约 `1e-6`。

这证明标定数值可用，也证明**不能在 cooperative pkl 上直接训练路端分支**：现有 `SPDE2EDataset.get_data_info(agent_name=...)` 会为路端读取路端相机信息，但 `get_ann_info(index)` 固定从 vehicle info 取 `gt_boxes`。若必须以 cooperative pkl 训练路端，需要将 3D boxes（中心、yaw、velocity）显式从 vehicle LiDAR 转至 infrastructure LiDAR，并在路端相机上可视化投影。

不过数据集中已有更安全的 A 单端训练路径：

- **D1 vehicle-only：** 使用 cooperative `spd_infos_temporal_train.pkl`，仅取 ego vehicle 数据和车端 GT；`vehicle-side/spd_infos_temporal_train.pkl` 当前为 0 字节，不能使用。
- **D2 infrastructure-only：** 使用 `infrastructure-side/spd_infos_temporal_train.pkl`；该文件有效（1,521 帧、26,056 个 valid GT，其中 21,220 个在路端 range），其 LiDAR pose 和 GT 是路端单端监督。
- **D3 cooperative：** 先加载 D2 的路端 detector 并冻结，再在 cooperative pkl 中仅做路端 query 推理、转换和车端融合；此时不对路端再次施加错误的 cooperative GT loss。

当前 `UniV2X` 工作区存在用户未提交的 config/转换/评测改动，且其 plugin 数据目录不受该仓库 Git 跟踪。为保证不覆盖本地实验，A 的下一步应在独立 config/工作目录中进行；在首次单 batch smoke 成功前禁止启动长训练。

#### E5-A / D1-D2 配置准备（2026-09-28）

已在 DrivewithVLM 中新增隔离配置（训练时从 `UniV2X/` 目录调用 `tools/train.py`）：

- `projects/configs/Robodrivevlm/univ2x_e5a_vehicle_det.py`：继承官方 vehicle tracker、关闭 `seg_head`，使用 cooperative detection pkl 的车端 GT，且将图片 loader root 修正为 cooperative root，避免 `vehicle-side/vehicle-side/...` 重复前缀。
- `projects/configs/Robodrivevlm/univ2x_e5a_infrastructure_det.py`：继承官方 infrastructure tracker、关闭 `seg_head`，使用 infrastructure-side 单端 GT。

路端与车端 config 均已完成 `build_dataset`（各 1,521 帧）及一个 5 帧时序样本的完整打包；样本含 `gt_bboxes_3d`、类别、instance ID、历史轨迹和自车框。车端 cooperative pkl 的图片重复前缀已由 config 修正。为隔离检测实验，已新增 `SPDE2EDetectionDataset`：不改变默认 `SPDE2EDataset`，仅移除 map/occupancy/planning 所需的 future target 与 temporal merge 字段。`spd_trajectory_api.py` 中一个重复 scene 循环也已修正，否则车端数据构建会重复预处理。GPU 4--7 未被本任务占用；下一项由用户执行 D1/D2 各 1 iteration 的 `forward_train`/loss smoke。

执行入口说明：E5-A 的两份 config 放在 DrivewithVLM 是为了与 VLM 实验配置集中管理，但它们实例化的是 `UniV2X` 的 `BEVFormerTrackHead`、`MultiAgent` 训练器和 detection-only dataset plugin；这些实现当前不在 DrivewithVLM 的 HuggingFace/LLaVA 训练脚本中。因此命令必须以 `UniV2X/` 为当前目录调用 `tools/train.py`（该目录也是 config 中相对 `plugin_dir=projects/mmdet3d_plugin/` 的解析基准），而 config 文件本身仍来自 DrivewithVLM。此前给出的多行命令被聊天界面断开为 `tools/` 和 `train.py`，不可直接复制；后续 smoke 命令应使用完整单行，或只在反斜杠后换行。

**D1/D2 单 iteration GPU smoke 命令（整行复制，不要在路径中换行）：**

```bash
# D1: vehicle-only；成功标准是完成 1 iteration 并出现 track.loss_*，不报错。
# PYTHONPATH 是必需的：直接执行 tools/train.py 时 Python 默认只把 tools/ 加入模块搜索路径。
cd /home/zzn/V2X_VLM/UniV2X && PYTHONPATH=/home/zzn/V2X_VLM/UniV2X CUDA_VISIBLE_DEVICES=4 /home/zzn/anaconda3/envs/univ2x/bin/python tools/train.py /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/projects/configs/Robodrivevlm/univ2x_e5a_vehicle_det.py --gpus 1 --no-validate --work-dir /tmp/e5a_vehicle_smoke --cfg-options data.train.is_debug=True data.train.len_debug=1 data.workers_per_gpu=0 total_epochs=1 runner.max_epochs=1 log_config.interval=1

# D2: infrastructure-only；仅在 D1 通过后执行。
cd /home/zzn/V2X_VLM/UniV2X && PYTHONPATH=/home/zzn/V2X_VLM/UniV2X CUDA_VISIBLE_DEVICES=4 /home/zzn/anaconda3/envs/univ2x/bin/python tools/train.py /home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/projects/configs/Robodrivevlm/univ2x_e5a_infrastructure_det.py --gpus 1 --no-validate --work-dir /tmp/e5a_infrastructure_smoke --cfg-options data.train.is_debug=True data.train.len_debug=1 data.workers_per_gpu=0 total_epochs=1 runner.max_epochs=1 log_config.interval=1
```

推荐的实施顺序（在 D0 前不改模型）：

1. **D0：坐标与监督闭环审计。** 对固定少量 token，输出车端 GT box、路端/车端相机投影、`VehLidar2InfLidar` 的方向及双向变换误差；确认 `gt_bboxes_3d`、类别映射、速度、有效标记和 10 类评测协议。此步必须有数值闭环和图像可视化，不能仅凭字段名称判断矩阵方向。
2. **D1：车端单端 3D 检测基线。** 使用当前官方 GT 与车端图像，在车端 LiDAR 坐标训练并评测一个独立 StreamPETR/BEVFormer 风格 head；只报告 3D 检测指标，不与规划损失耦合。目的是先证明数据、投影和 label 协议正确。
3. **D2：路端单端检测基线。** 使用路端图像和路端自身 `pc_range` 训练/加载路端 detector，输出路端 query；同样单独评测。若 D1/D2 任一投影/检测失败，不进入融合。
4. **D3：复用 UniV2X 的 query 融合。** 严格复用并单测 `veh2inf_rt` 的方向：路端 ref point 反归一化 → inf-to-vehicle 刚体变换 → 车端归一化 → 匹配/融合。输出与评测均转换/保留在车端坐标。先比较 vehicle-only 与 cooperative detection，验证遮挡区域是否有真实增益。
5. **D4：检测到规划的融合。** 将融合后的、带位置和置信度的 detection queries 通过投影层输入未来的 trajectory head；训练初期冻结检测分支，只加规划 loss，稳定后再以小权重联合微调。此步骤依赖 E4 的连续轨迹 head，不能直接将 boxes 拼成文本替代检测监督。

每一步的准入条件：D0 变换闭环通过；D1/D2 分别有可复现实验和检测报告；D3 相对 vehicle-only 的检测结果不退化并有遮挡场景收益；D4 同时报轨迹、碰撞和检测指标，检测性能不得因联合训练出现不可接受退化。

### V2 physics validation 运行状态（2026-09-28）

6 个 checkpoint 的 validation sweep 已完成，汇总在 `evaluation_results/meta_v2_physics_seed42_4gpu_val/sweep_summary.json`，可复现选择记录在 `best_checkpoint_selection.json`。选择规则为 coverage ≥ 99%，再按最小 validation mean L2（末点 L2 仅作平局裁决）；所有 checkpoint coverage 均为 100%（157/157），最终选中 **checkpoint-932**：mean L2 **1.1442 m**、4.5 s 末点 L2 **3.2208 m**、横向/纵向文本 macro-F1 **49.57% / 63.77%**、横向/纵向文本与轨迹动作一致率 **98.09% / 89.17%**、目标末速度 MAE **0.9523 m/s**、平均加速度 MAE **0.2405 m/s²**。

这表示：模型的轨迹文本生成完整可解析（157/157）；在该 validation split 上，轨迹误差从近端逐步增长到 4.5 s 的 3.22 m，且有 30/157 条末点误差超过 5 m、2/157 超过 10 m。横向的总体 accuracy 虽为 81.53%，但两种稀有变道类别均未命中（macro-F1 只有 49.57%），不能被 KEEP_LANE 的高召回掩盖；纵向对 ACCELERATE、DECELERATE 较可靠，但 KEEP_SPEED 召回只有 36.67%，并且 validation split 没有 STOP 样本，故无法评估 STOP。checkpoint-1398 的动作指标略高（横向 49.85%、纵向 64.75%），但其 mean/末点 L2 均较差（1.1987/3.3595 m）；按既定以轨迹主指标选型的规则不替换 checkpoint-932。

截至本次检查，`evaluation_results/meta_v2_physics_seed42_4gpu_test/` 已创建但尚未发现 test diagnostics，且未看到正在运行的对应 test 进程；因此此处只记录 validation 结论，不把它当作 test 成绩或 E0 的直接胜负结论。

## 1. 背景与当前结论

当前系统使用 UniV2X/SPD 生成 V2X 数据，并通过 `MMdrive_v2x.py` 完成双视角视觉输入、场景文本提示和 9 点轨迹生成。已有评测表明，当前 checkpoint 的主要问题不是全局坐标系错误，而是长时域误差累积、动作类别偏科和少量文本解码失控。

当前代表性结果：

| 指标 | 当前值 |
|---|---:|
| 有效预测数 | 672 |
| 全 9 步平均 L2 | 约 1.42 m |
| 4.5 s 平均 L2 | 约 4.23 m |
| 4.5 s L2 P90 | 约 8.17 m |
| 任意时间步碰撞样本率 | 约 6.25% |
| Meta Action 准确率 | 约 48.36% |
| 非法轨迹率 | 约 0.30% |

关键观察：

- 0.5 s 平均误差仅约 0.016 m，说明模型能从当前自车状态拟合第一个点。
- 误差随预测时间单调增大，说明当前文本式绝对坐标生成缺少运动学和长期约束。
- `011180`、`011210` 出现累计 X 倒退，是典型的生成解码失控。
- 变道、快速加减速等少数动作类别准确率接近 0，动作分布明显不均衡。
- 当前输入缺少明确导航意图、route 和结构化车道信息，仅凭画面与历史状态无法唯一确定路口转向。
- 旧训练没有 validation checkpoint selection，最终 epoch 不一定是可靠 baseline。

## 2. 总体目标与验收指标

下一阶段目标是先建立可信 baseline，再逐步将轨迹预测从纯文本生成升级为带结构化道路与目标信息的连续回归规划。

所有实验统一报告：

- 预测覆盖率和解析失败率。
- 0.5/1/2/3/4/4.5 s 的 mean、median、P90 L2。
- 全 9 步平均 L2 与 4.5 s FDE。
- 任意时间步碰撞样本率和逐时间步碰撞率。
- 非法轨迹率及具体原因。
- Meta Action accuracy、macro-F1 和混淆矩阵。
- 按动作类别统计 L2、FDE、碰撞率和非法率。

阶段性最低验收标准：

1. 预测覆盖率不低于 99%。
2. 非法轨迹率降至 0，且不通过删除预测样本实现。
3. 相比选出的旧版 baseline，全 9 步平均 L2 至少下降 15%。
4. 4.5 s L2/FDE 至少下降 15%，碰撞率不得恶化。
5. Meta Action macro-F1 至少提升 10 个百分点。

## 3. 阶段一：确定旧版可靠 baseline

### 3.1 批量评测

在旧版匹配测试集 `test_dt_resampled_4.json` 上重新推理：

```text
checkpoint-388
checkpoint-776
checkpoint-1164
checkpoint-1552
checkpoint-1940
checkpoint-2328
```

使用 `tools/eval/sweep_checkpoints.py` 生成每个 checkpoint 的预测、碰撞结果、非法轨迹报告和动作指标。

baseline 选择规则：

1. 覆盖率必须达到 99%。
2. 优先选择全 9 步平均 L2 最低的 checkpoint。
3. L2 接近时依次比较碰撞率和非法轨迹率。
4. 不使用最后 epoch 作为默认 baseline，也不删除非法样本后重新排名。

### 3.2 固化实验信息

选出 baseline 后记录：

- checkpoint 绝对路径和训练 step/epoch。
- Git commit、配置文件和 test JSON。
- 完整推理命令与 GPU 数量。
- 所有汇总指标和失败 token。

该 baseline 将作为后续所有优化实验的唯一对照组。旧数据实验与 causal-history 实验必须分表报告，禁止跨数据版本直接比较。

## 4. 阶段二：修复数据、Prompt 与训练选优

### 4.1 使用因果历史轨迹

新训练切换为：

```text
data/Planning/train_v2x_causal_history.json
data/Planning/test_v2x_causal_history.json
```

要求：

- train/test 均只能使用当前时间戳之前的真实 pose。
- 历史不足时使用明确 mask，并保留样本，不用未来轨迹反推历史。
- 训练与测试使用完全一致的历史采样间隔和坐标系。
- 在训练前统计历史点有效率、连续性、速度和加速度分布。

### 4.2 简化 Meta Action

将一个复合字符串拆成两个监督字段：

```text
Lateral Action:
KEEP_LANE / TURN_LEFT / TURN_RIGHT /
LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT

Longitudinal Action:
ACCELERATE / KEEP_SPEED / DECELERATE / STOP
```

不再将 `QUICK` 作为独立类别，变化强度由连续速度或加速度表示。保留旧 Meta Action 解析器仅用于历史结果兼容。

### 4.3 类别均衡

- 统计 lateral/longitudinal 联合分布和每类轨迹数量。
- 对少数类使用 weighted sampler 或过采样。
- train/validation 按 scene/sequence 划分，禁止相邻时间帧跨 split。
- 单独报告转弯、变道、停车和快速速度变化场景。

### 4.4 Validation 选优

训练时每个 epoch 在固定 validation split 上评测：

- 全 9 步平均 L2。
- 4.5 s FDE。
- 碰撞率。
- 非法轨迹率。
- Meta Action macro-F1。

主选优指标为全 9 步平均 L2；碰撞率、非法率作为约束指标。保存 best checkpoint 和 last checkpoint，但测试集只用于最终一次报告。

## 5. 阶段三：轨迹表示与连续回归头

### 5.1 文本轨迹的短期改造

在连续回归头完成前，将文本输出从累计绝对坐标改成相邻位移：

```text
Delta Trajectory:
[(dx1,dy1), ..., (dx9,dy9)]
```

解析后通过累加获得累计轨迹。生成结果必须经过：

- 固定 9×2 shape 检查。
- finite 检查。
- 速度、加速度和纵向倒退检查。
- 曲率和平滑性检查。

非法结果保留原始输出并报告，同时可单独给出恒速度 fallback 指标；fallback 指标不得替代模型原始指标。

### 5.2 轨迹回归头

最终取消用 LLM token 直接表示浮点轨迹。LLM 负责场景推理和动作意图，轨迹由连续 head 输出：

```text
视觉/V2X BEV 特征
        +
LLM planning hidden state
        +
自车状态与历史轨迹 embedding
        ↓
Trajectory Head
        ↓
[B, 9, 2]
```

建议损失：

```text
L = 1.0 * L_traj
  + 0.5 * L_velocity
  + 0.2 * L_acceleration
  + 0.1 * L_smoothness
  + 1.0 * L_collision
  + 0.5 * L_lane
```

其中 `L_traj` 使用 SmoothL1；其余权重根据实际 loss 数值尺度调整，确保任何单项 loss 不长期压制其他任务。

## 6. 阶段四：车道线与 3D 检测辅助任务

### 6.1 推荐结构

```text
车端图像 + 路侧图像
          ↓
    多视角视觉编码
          ↓
      V2X/BEV 融合
          ↓
 ┌────────┼──────────┐
 │        │          │
3D检测头 车道线头  VLM场景推理
 │        │          │
 └────────┴────┬─────┘
               ↓
          轨迹回归头
               ↓
          9×2 未来轨迹
```

### 6.2 车道线 head

第一版优先实现：

- 可行驶区域分割。
- 车道边界/道路边缘分割。
- 停止线监督。

稳定后升级为 lane queries + polyline 控制点，并让轨迹 query 对 lane queries 做 cross-attention。车道监督全部使用当前车辆坐标系：X 向前、Y 向左。

### 6.3 3D 检测 head

输出至少包含：

- 类别。
- 车辆坐标系下的 3D 中心。
- 长宽高、朝向和速度。
- 置信度及有效 mask。

规划模块应直接读取 detection queries，不再把全部目标转换成长文本。文本 prompt 只保留高风险或与 route 相关的结构化摘要。

### 6.4 导航与 Route

车道线只能描述道路结构，不能决定车辆驾驶意图。规划输入还应增加：

- `GO_STRAIGHT / TURN_LEFT / TURN_RIGHT` 导航命令，或
- 车辆坐标系下的 route polyline/目标车道中心线。

没有导航或 route 的实验必须单独标注为无导航设置，避免把不可观测意图错误归因于模型能力。

## 7. 实验矩阵

按单变量原则依次执行：

| 实验 | causal history | 动作简化/均衡 | route/lane | 回归头 | 3D det | 目的 |
|---|---:|---:|---:|---:|---:|---|
| E0 | 否 | 否 | 否 | 否 | 否 | 旧版可靠 baseline |
| E1 | 是 | 否 | 否 | 否 | 否 | 验证历史轨迹修复 |
| E2 | 是 | 是 | 否 | 否 | 否 | 验证动作监督与均衡 |
| E3 | 是 | 是 | 是 | 否 | 否 | 验证道路/导航信息 |
| E4 | 是 | 是 | 是 | 是 | 否 | 验证连续轨迹回归 |
| E5 | 是 | 是 | 是 | 是 | 是 | 完整 V2X-VLM 多任务模型 |

每个实验必须使用相同 scene split、相同评测脚本和相同 occupancy GT。若数据版本变化，需要重新生成对应 baseline。

## 8. 推荐实施顺序

1. 完成 6-checkpoint sweep，确定 E0 baseline。
2. 固化 causal-history 数据统计与 scene-level validation split。
3. 简化动作标签并实现类别均衡。
4. 训练 E1/E2，确认数据与动作改造有效。
5. 加入导航/route 或基础车道监督，训练 E3。
6. 实现连续轨迹回归头及运动学/碰撞 loss，训练 E4。
7. 最后加入 3D 检测 head，训练 E5。

只有当前一阶段通过验收标准后，才进入下一阶段，避免同时改变数据、Prompt、网络结构和 loss 后无法定位收益来源。
