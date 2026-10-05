# DrivewithVLM 项目协作说明

## 每个新会话的初始化阅读

项目根目录：`/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM`。
用户将 `docs/` 作为项目知识库；先读项目资料，再提出模型路线、修改代码或解释评测。

1. 完整读取 `docs/PROJECT_CONTEXT.md`，了解文档索引、当前协议和已知历史结果。
2. 列出 `docs/` 中的 Markdown 文件，查看各文档标题和更新说明，识别新增资料。
3. 读取 `docs/roadside_agent_token_fusion_plan.md` 的第1～3节及历史部分末尾修订，
   `docs/v2x_vlm_optimization_plan.md` 的“实施进展”部分，以及完整
   `docs/v2x_vlm_next_steps_20260930.md`、`docs/readme.md`。
   不要只读方案开头：实际实现和用户后续决定可能记录在末尾。
4. 连续规划任务还要完整读取 `tools/continuous/README.md`。
5. 根据当前任务完整读取相关方案章节；研究定位和基线设计读取研究报告（7），
   必要时追溯报告（6）与两份优化计划。论文核查读取 `docs/references/` 的索引及
   对应原文；无需在每次启动时逐篇解析所有参考 PDF。
6. 首次项目回复简要说明已加载的主要资料；未读取的原文不要声称已读。

阅读可分段进行；工具输出被截断时继续读取所需部分。索引是导航，不替代原文。
初始化阅读是行为要求，并不意味着 docs 全目录会由客户端自动注入上下文。

## 文档与事实的优先级

- 当前用户指令优先。文档之间冲突时，优先采用明确的后续用户协议修订和可核实产物，
  再参考同主题最新实施记录；长期调研建议不能覆盖已确认的执行协议。
- `roadside_agent_token_fusion_plan.md` 末尾“当前执行协议：先用车端 GT 验证”
  已更新早期 image-only 方案；`v2x_vlm_next_steps_20260930.md` 中“head 未接入”等
  状态也要与后续 `tools/continuous/` 实现记录核对。
- 文档结果注明日期、数据版本、输入来源和 split。当前训练状态、checkpoint 是否存在、
  test 是否完成，必须读取对应产物核实；不能把 2026-09-30 的记录当作实时状态。
- 完成实现、训练、评测或分析后，更新对应进展文档的状态、证据路径和下一步；
  若修改当前协议或文档组织，同时更新 `docs/PROJECT_CONTEXT.md`。

## 当前研究与实验约定

- 主线：可学习的连续 ego planner → Oracle 路侧几何价值 → Detector 路侧输入 →
  有证据后再扩展对象交互、VLM 内注入、通信鲁棒性与 Flow。
- 当前开发方案先比较I1常规视觉、I2视觉carrier queries、I3融合semantic tokens三种VLM输入。所有RSU信息必须进入VLM，禁止planner侧RSU拼接/cross-attention旁路。旧M1已实现但只保留为历史。先验收输入接口，用同deterministic head评测，再研究VAE＋GRU。
- 最新实施顺序见融合计划第3.2节：先独立双端GT共享ObjectEncoder/source与独立StateEncoder，固定范围验收，再可变范围和规划相关性排序，后I2/I3。I2/I3主方案保留原始车端图像，query-only为消融；OmniDrive本地use_lidar=False，LiDAR坐标/深度采样不得写成实测点云输入。第2节接口用户要求保持原文；旧设想删除，系统/代码记录保留第4节。
- 当前正式车端输入为图像＋独立 vehicle-side LiDAR GT 对象文本＋自车状态/因果历史＋
  固定规划提示，数据版本 `continuous_v2_ego_gt`，属于 Oracle ego perception。
  Ego-only 仅指无路侧输入。F0/F1/F2 固定相同车端来源和 prompt。
- F0 为 zero_rsu，F1 为独立路侧 GT，F2 为冻结 D2 预测。F0 未通过正式验收时，
  按现有协议先修 planner；不能用极小 smoke 宣称正式收益。
- 不将 cooperative GT 整体当成车端或路端可观测对象；不得通过改 source 标签
  将 GT 冒充 detector。未来轨迹、由其产生的动作/CoT、未来估计速度不能进入观测上下文。
- 坐标统一为当前 ego LiDAR XY（米，X 前、Y 左）；显式核对变换方向、lwh/wlh、
  yaw、速度和时间。首版路侧 velocity-disabled；age 仅元数据，未做运动补偿。
- 主规划指标为 L2@1/2/3s 及三者平均，3s FDE 与 4.5s FDE 分开；
  全9步 mean L2 不能当作三时域平均。报告 coverage、缺失/非法预测和有效样本数。
- validation 用于选型，test 用于最终报告；数据版本、scene split、GT/Detector 来源、
  route、训练预算或预训练不同的结果不能直接宣称公平胜负。缺失 collision checker
  时指标为未评测/null，不能记为零碰撞。

## 执行入口与实验产物

- 连续规划：项目根目录，参考 `tools/continuous/README.md`，常用解释器
  `/home/zzn/anaconda3/envs/robollm/bin/python`。
- D1/D2 检测：从 `/home/zzn/V2X_VLM/UniV2X` 调用其 `tools/train.py` / `tools/test.py`，
  config 位于本项目 `projects/configs/Robodrivevlm/univ2x_e5a_*_det.py`，
  常用解释器 `/home/zzn/anaconda3/envs/univ2x/bin/python`。
  参考 `docs/readme.md`；当前 test 配置对应 SPD validation split，名称不能据此改成 test 成绩。
- 执行用户要求的训练或评测前，核对数据、权重、GPU 占用、现有进程和输出路径；
  文档中的命令示例不代表本次任务要求立即运行。新实验使用独立输出目录，保留旧数据、
  checkpoint 和用户未提交的改动；不抢占或终止其他训练。
- 根据实际修改做必要验证；工程 smoke、数值稳定性、检测 AP 和规划收益分别报告。

## 已选定的后续升级方向（2026-10-04）

涉及后续planner调研/开发时，读取`docs/next_plan_image_spatial_state_vae_20261004.md`。用户选择保留原始image tokens，增加空间/时间适配监督、数值state编码、任务token与条件VAE＋GRU。先做E2 LoRA冻结特征＋同MLP和state/readout对照。该方向目前只有计划，不代表代码已升级或授权在启动会话时自动训练；旧M1结果继续保留。

最新修订优先：用户要求三种输入都尝试，先想清VLM拼接；`roadside_agent_token_fusion_plan.md`第1～3节为当前依据。自车state先作为VLM输入tokens；此前state decoder旁路仅待定消融。不要将旧M1或任何RSU planner旁路重新当成当前方案。

最新用户修订（2026-10-04）：首阶段保持VLM自回归生成轨迹文本，不以连续MLP或VAE head为前置。已有检测结果来自官方stage-1权重、675帧；与当前规划train/val无重叠，覆盖test的ego480帧和已有RSU消息475帧。详见docs/d1_d2_existing_results_audit_20261004.md；生成接口须另适配soft tokens，不能将online连续头代码当作已支持generate。

2026-10-04数据修复：已恢复完整车端PKL、另导双端无补框Native版本；新版Oracle数据`continuous_v3_native_gt`（929/157/480）从两端原生JSON读取对象，prepare不再把路端PKL补框当观测。旧v2及checkpoint保留历史。详见docs/independent_observation_repair_20261004.md；原生AP evaluator未重评，不能将数据修复当作检测/规划收益。

最新补偿/关联修订（2026-10-04）：前三项已实现于tools/association/，详见docs/native_association_motion_audit_20261004.md。原生历史因果恒速补偿和A/B/C关联对照已跑完整train/val；旧prepare/planner仍保持其版本行为，禁止把新宽候选JSON直接交给旧planner。正式关联阈值和T1/S1输入导出尚未完成，不在会话初始化时自动训练。

2026-10-04尺寸稳定/融合实现：tools/association/stable_fusion.py及配套入口已完成A0–A4全train/val与冻结后test，28项测试通过；默认冻结A1。软关联高可信和尺寸平均未过验收，cor继承ego尺寸不能证明物理精度。最新证据见evaluation_results/stable_fusion_v1_20261004/report.md与docs/causal_size_stabilization_fusion_plan_20261004.md。尚未导出T1/S1输入或运行VLM训练，禁止从离线算法完成推断生成接口完成。


2026-10-04关联放宽V2：已实现`tools/association/relaxed_fusion.py`与`evaluate_relaxed.py`，完整train929/val157，未使用test。A1/L1/L2三组统一原始尺寸；L1取消距离减半；L2对age≤0.1s、cost≤0.45且BEV IoU≥0.1的未知运动对象允许配对，保留朝向/歧义约束。val召回A1=88.46%、L1=79.02%、L2=80.65%，可判定错配均1条。新规则改善旧A3的保守性但未胜过A1，默认仍A1；未替换VLM缓存。结果见`data/Planning/relaxed_fusion_v2_trainval_20261004/report.md`，关联相关21项测试（6项新增）通过。

2026-10-04 T1/S1自回归接口已实现于tools/object_vlm，缓存dual_object_a1_v1_20261004，31项CPU测试通过；真实7B smoke因显存占用未通过，正式训练未启动。用户最新指令是只提供脚本，由用户运行，禁止自动启动GPU实验或等待队列。最新状态读docs/t1_s1_object_vlm_implementation_20261004.md与tools/object_vlm/README.md。

2026-10-04最新实测：work_dirs/T1_S1_objects_4gpu_smoke_20261004中T1/S1真实7B单卡梯度/对象敏感性/重载生成检查均通过，双组0–3四卡9train/8val、3更新、分片验证与保存均通过；34项相关CPU测试通过。正式6epoch尚未启动。短probe严格轨迹解析0/8，不能作为规划效果。正式脚本为ACTION=train TRAIN_GPUS=0,1,2,3 RUN_ROOT=work_dirs/T1_S1_objects_4gpu_smoke_20261004 bash tools/object_vlm/run_t1_s1.sh；T1先四卡6epoch，再S1四卡6epoch，仍由用户启动。结果见该目录report.md和smoke_summary.json。
