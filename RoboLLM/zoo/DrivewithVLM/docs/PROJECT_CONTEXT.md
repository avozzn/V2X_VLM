# 项目上下文与文档索引

整理日期：2026-10-04。历史结果主要来自2026-09-28～09-30；
2026-10-04已核查正式F0产物并在CPU重载best复算validation，详见
`f0_training_audit_20261004.md`。其他历史实验未在本次全部复核。

## 项目目标与当前执行协议

项目研究 DAIR-V2X / V2X-Seq-SPD 上的车路协同自车轨迹规划。
原开题路线为路侧稀疏 Query、跨视域对齐、VLM 推理及文本坐标；后续路线收敛到
结构化路侧 Agent 表示与连续轨迹 head，并逐项验证额外路侧信息的价值。

最新输出协议：首阶段保持VLM自回归直接生成轨迹文本，先研究独立双端感知/soft-token输入；连续MLP/VAE头推迟。既有E2直接生成结果不代表已接入Detector。检测已有结果及split审计见`d1_d2_existing_results_audit_20261004.md`。

当前设计协议以`roadside_agent_token_fusion_plan.md`第1、3节为准：两个目标是双端独立感知/对象选择/时空对齐/空间query，以及独立StateEncoder/因果历史鲁棒性。先双端GT soft tokens的I1，固定范围与数值接口验收后比较可变范围，再I2视觉carrier、I3融合semantic；主方案保留车端图像，query-only另作消融。所有路侧信息必须经VLM，禁止planner侧RSU旁路。先固定语言生成后端；后续再比较连续head/VAE＋GRU。新版I1路侧在线输入核心已实现并通过CPU接口检查；真实7B和正式训练未运行。第3.2节列初步4～6周主线排期，Detector/BEV/map扩展另排；不是已启动实验或完成承诺。

下表记录**已实现的旧M1输入与产物**，不代表当前开发指令；`tools/continuous/README.md`包含新版在线入口与旧代码说明。

| 项目 | 旧M1已实现合同（新版需另开发） |
|---|---|
| 车端输入 | 车端图像、独立 vehicle-side 原生 LiDAR GT 对象文本、自车尺寸/速度/加速度/因果历史、固定规划提示 |
| 感知 setting | Oracle ego perception；车端 GT 尚未限定前向相机 FOV，不是纯图像或真实 detector 组 |
| 数据 / prompt | `data/Planning/continuous_v2_ego_gt/`；`ego-perception-observation-v2` |
| split | train/val/test=929/157/480，沿用原 token split；内部子集，与官方论文协议需另作核对 |
| VLM readout | 冻结 LLaVA 的 final-normalized、图像合并后最后有效上下文 token；不读取 assistant GT 答案；`[PLAN]` 后置 |
| 融合 | M1：13→256→256 Agent MLP、masked mean、拼接 h_ego 与 h_rsu；三组 head 均为 D+256 输入 |
| 轨迹 | 9×2 displacement，cumsum 为 waypoint；唯一 masked cumulative-waypoint Smooth-L1 |
| F0 / F1 / F2 | 路侧分别为 zero / 独立 RSU GT / 冻结 D2 预测；同构、同初始化规则、分别独立训练 |
| 对象预算 | 车端50m ROI、最多32对象；路侧50m ROI、K=16 |
| 路侧限制 | 13D 中 vx/vy 首版禁用、置零；读取截至ego时刻可得消息，可能有约2.2s消息年龄；未运动补偿，age未输入encoder |
| 下一步顺序 | 核验完整缓存与正式F0验收；通过后F1；D2验证与预测导出就绪后F2；随后再考虑cross-attention等升级 |

车端 GT 从 `vehicle-side/data_info.json` 指定的原生 `label/lidar` 读取，核对 ego
LiDAR timestamp；缺失对象速度/置信度在文本中为 unknown。旧 cooperative PKL
的 `perception_text(info)` 不能整体搬进 F0 冒充车端独立观测。
旧 continuous_v1 cache 仅保留为 smoke，不能用于修订后的正式协议。

2026-10-04独立3D观测源码核查：两库单端转换均有未来观测补框与下一帧GT速度；新版Oracle双端直接读取当帧原生JSON。已修复prepare.py路端对象来源并导出`continuous_v3_native_gt`（929/157/480）；车端损坏PKL已备份并恢复完整1521train/675val，另建两端Native版本，无补框。原始/New数据引用全部存在非空、JSON可解析，无漏下载证据，故障在派生转换/保存。详见`independent_observation_repair_20261004.md`。此前路端val的1124补框仅为历史来源证据，不代表新版Oracle输入。UniMM单端MoE默认/显式关闭的结论保留，未重算AP或训练。

## 评测记录及解释边界

3D来源历史复查：路端train1638个/val1124个补框已全部复现。补框作为离线监督/评测GT不自动构成泄漏，限制针对直接输入VLM的当前观测。车端旧20帧val是完整列表的精确前缀，历史中断原因未知。随后已恢复train保存并改成完整转换后原子发布，完成重导；完整证据见修复报告。

2026-10-04检测mAP补充：已有官方D1/D2预测的十类mAP=3.09%/12.19%，实际GT只有car/bicycle/pedestrian，七个零类拉低分母；三类AP诊断平均=10.29%/40.62%，不是官方成绩。两份权重MD5与官方一致。官方TRAIN_EVAL表报告协同AMOTA/地图IoU，无单端stage-1检测mAP；车端有效类别仍弱，GT覆盖/可观测性等原因尚待核查。详见d1_d2_existing_results_audit_20261004.md §2.1。

| 实验 | 文档记录 | 解释边界 |
|---|---|---|
| E0旧文本baseline | checkpoint-1552；675/675，9步mean L2=1.4264m，4.5s L2=3.4648m，collision-any=7.11% | 旧数据协议；不能直接与新V2 split比胜负；来源为优化计划进展表 |
| E2 V2 physics validation | checkpoint-932；157/157，9步mean L2=1.1442m，4.5s FDE=3.2208m；横/纵macro-F1=49.57%/63.77% | validation选型结果，含GT perception文本，不能证明纯双图像或路端独立收益 |
| E2主时域换算 | L2@1/2/3s=0.0924/0.4797/1.2398m，三者平均0.6040m | 来自9月30日下一步文档；与9步mean L2是不同指标 |
| E2 test | 文档记为待最终报告 | 下次涉及当前状态必须检查真实产物，不能凭旧记录断言仍未完成 |
| D1/D2 | 已有数值稳定性smoke、训练/评测命令及官方checkpoint验证入口 | smoke和loss不能替代mAP/NDS、AMOTA/AMOTP；需核验当前日志与结果 |
| F0车端GT极小smoke | 2条train/2条val、30epoch；best主时域平均L2=1.824m，恒速=0.554m | 仅工程链路验证，未过正式F0门槛，不代表完整数据泛化或协同收益 |
| F0正式seed42（10月4日已复核） | 929train/157val，100epoch于10月2日完成；best epoch78，主L2=3.9697m，4.5s FDE=9.8169m，coverage157/157 | 工程成功，未过F0验收；同val恒速1.4660m、均值3.8762m；先诊断F0再F1 |

结果路径线索：

- E0：`evaluation_results/E0_fixed/gpu3/checkpoint-1552/diagnostics.json`。
- E2：`evaluation_results/meta_v2_physics_seed42_4gpu_val/` 下的
  `sweep_summary.json`、`best_checkpoint_selection.json`。
- F0旧smoke：`work_dirs/continuous_F0_smoke_20260930/`。
- F0车端GT smoke：`work_dirs/continuous_F0_ego_gt_smoke_20260930/`。
- D1/D2训练：`work_dirs/e5a_d1_d2_q3_full_8gpu_e30/`。
- 最新正式实验结果优先读取真实 `run.json`、`best_metrics.json`、`history.jsonl`、
  diagnostics或eval.log，并记录具体split、输入来源与命令。

## 文档地图

最新双端输入设计：车端/路端独立 GT 共享对象编码器 + 来源 embedding，自车状态/历史使用独立 StateEncoder，全部在 VLM 前输入；固定/速度/规划相关性范围分阶段实验。见 `object_soft_tokens_adaptive_roi_plan_20261004.md`。当前代码仍只有 RSU soft tokens，新增设计尚未实现或训练。

下面路径均相对于本 docs 目录。

| 文件 | 内容与何时读取 |
|---|---|
| `readme.md` | 实际命令：官方D1/D2验证、训练、连续数值评测、串行脚本；运行前必读 |
| `f0_training_audit_20261004.md` | 正式F0完成证据、最佳epoch、基线比较、CPU重载和统一评测；当前F0状态优先参考 |
| `next_plan_image_spatial_state_vae_20261004.md` | 用户选定的后续方向：保留图像tokens、学习时空对齐、显式state、任务token与VAE＋GRU；阶段计划，尚未实现或训练 |
| `trajectory_head_training_research_20261004.md` | 25份参考PDF筛查、AURORA/Omni等轨迹头训练核查、状态shortcut与F0升级建议；建议尚未改变执行协议 |
| `roadside_agent_token_fusion_plan.md` | 第1节两个目标/四个子目标及参考代码；第2节旧共用路侧接口原文保留；第3节实施方案/排期/实验；第4节系统代码和真实历史记录。旧M1～M8设想已删，实际代码产物保留 |
| `independent_3d_observation_review_20261004.md` | UniV2X/UniMM单端3D架构、实际MoE开关、原生标签与PKL补框统计、独立Oracle/Detector缓存及评测实施顺序 |
| `independent_observation_repair_20261004.md` | DAIR-V2X原始数据引用完整性、PKL转换/原子保存修复、完整双端原生版本与规划v3导出、备份路径和13项测试 |
| `detector_next_steps_20261004.md` | 检测与VLM双线接续；D1 epoch30数值有限、D2 epoch5起权重损坏/AMP scale归零，独立评测JSON、统一AP与冻结缓存优先级 |
| `d1_d2_native_eval_script_20261004.md` | 官方双端独立原生GT/5帧评测已完成675帧/端；D1/D2三类mAP=23.20%/41.33%，十类=6.96%/12.40%，完整场景分四卡与结果路径 |
| `detector_optimization_research_20261004.md` | TF-ETCP来源待核验、与NMS-free检测头适配分析；Circle NMS已做CPU探索对照，D1/D2三类mAP=23.84%/42.56%，后续因果memory与深度监督建议 |
| `bev256_detector_experiment_20261004.md` | BEV200→256配置/官方参数插值已实现，双端实际模型CPU加载通过；用户运行推理/AP，尚未微调或报告新精度 |
| `v2x_vlm_next_steps_20260930.md` | 当日代码/结果审查，输入来源、因果readout、数值评测和开发边界；部分实施状态已被后续更新 |
| `v2x_vlm_optimization_plan.md` | E0～E5历史路线、进展表、检测坐标审计、E2选型；先读实施进展，再读相关章节 |
| `v2x_vlm_model_optimization_and_experiment_plan.md` | 9月28日长期模型/实验规划，基线、通信、可选闭环；作为设计背景 |
| `deep-research-report (7).md` | 较新的中文研究调研、创新定位、Agent表示、外部baseline、消融与路线；相关工作原文需核查 |
| `deep-research-report (6).md` | 较早英文调研、agent概念、数据、指标与里程碑；用于追溯研究依据 |
| `开题报告-12.19.pdf` | 原始研究问题与论文背景；讨论开题、论文结构时读取相关部分 |
| `references/README.md` | 下载论文与来源目录索引；涉及论文时核对真实文件名 |
| `references/references.md` | 引用信息、原文链接与摘要；不替代论文原文，历史路径可能与实际下载文件不同 |
| `../tools/continuous/README.md` | 已实现连续planner的输入限制、运行命令、缓存/checkpoint与验证；不是仅有建议的设计稿 |

参考论文PDF按 arxiv/openaccess/aaai/neurips/ecva/local 分类保存。
研究报告仍包含部分旧 `turn...` 引用占位，不能直接作为可追溯引用或新颖性证据。
参考文献README的PDF计数文字存在不一致；按目录实际文件核对。

## 保持上下文准确

每次会话先查看是否出现新增文档和更晚修订。协议发生变化时同步修改此索引，
让“当前协议”和“历史实验”保持区分。这里的下一步是文档记录，执行前先核对是否
已有新的完整缓存、训练或评测结果。记录“已实现”“smoke通过”“正式评测通过”时
分别给出证据；不从其他阶段的成功推断本阶段完成。

10月4日补充诊断：当前F0使用冻结7B LLaVA的缓存hidden，100epoch只训练head，
没有LoRA。best train/val主L2=2.5155/3.9697m，last=1.9753/4.3042m。
速度/历史仅经文本进入末token表示，无数值旁路；train-only数值state线性探针
在val得到0.6312m，而冻结hidden线性探针约3.33～4.06m。
这些支持信息通路与跨scene泛化不足，未确定唯一根因；详情见F0审查报告。

2026-10-04用户进一步选择保留原始图像tokens、借鉴spatial queries并增加可监督的路端→车端时空适配、显式数值state与VAE＋GRU。详见next_plan_image_spatial_state_vae_20261004.md。这是后续升级方向；旧M1协议/结果保留，新版尚未开发。本轮只写接续计划。

最新用户硬约束：RSU不得作为VLM后的planner条件，三种输入接口均需尝试；旧M1只保留历史。参见融合方案第3节与下一轮计划第13节。

紧急接口修复：`tools/continuous/online_model.py`＋`online.py`已实现RSU-before-VLM，禁用旧M1 oracle/detector旁路，旧F0保留重载。15项CPU测试通过；真实7B待smoke。I1其他完整功能/I2/I3/LoRA联合训练/VAE仍待开发，详细范围见tools/continuous/README.md。

2026-10-04最新首轮范围：仅T1独立双端GT文本 vs S1同对象soft tokens，VLM直接生成；ego-only/动态范围/StateEncoder升级暂缓，最小对照两组保持相同state/history文本。当前运行MMdrive_v2x.py为双图像+state/history，无对象文本，不能计T1。见running_v2x_t1_eligibility_20261004.md及主方案3.2.1。

2026-10-04最新补偿/关联进展：`tools/association/`已完成原生双端参考身份审计、同track过去历史平面恒速补偿、A/B/C关联扫描；train929/val157完整CPU实验，7项测试通过。详见`native_association_motion_audit_20261004.md`。新实验接口独立于旧prepare/planner缓存；尚未选正式去重阈值、未接入T1/S1训练。旧velocity-disabled/no-motion记录是历史版本，不能覆盖本轮新方案。第2节接口按用户要求保持原文。

2026-10-04最新实现：`tools/association/stable_fusion.py`及评估/冻结/报告入口完成双端因果尺寸稳定、软IoU/朝向关联、高可信判断、尺寸选择/平均。A0–A4完整train929/val157、冻结后test480及28项测试通过；算法无cor标签依赖。val补偿召回67.60%→88.46%，ego/RSU尺寸波动降23.69%/33.70%；软关联高可信召回63.87%，条件平均未改善cor一致性，冻结保留A1。融合评估cor尺寸全部继承ego，不能证明物理尺寸准确性。完整报告：`../evaluation_results/stable_fusion_v1_20261004/report.md`，实现状态：`causal_size_stabilization_fusion_plan_20261004.md`。未做ROI/top-K、T1/S1输入导出、VLM训练或检测AP；旧缓存行为保持历史版本。


2026-10-04关联放宽V2：已实现`tools/association/relaxed_fusion.py`与`evaluate_relaxed.py`，完整train929/val157，未使用test。A1/L1/L2三组统一原始尺寸；L1取消距离减半；L2对age≤0.1s、cost≤0.45且BEV IoU≥0.1的未知运动对象允许配对，保留朝向/歧义约束。val召回A1=88.46%、L1=79.02%、L2=80.65%，可判定错配均1条。新规则改善旧A3的保守性但未胜过A1，默认仍A1；未替换VLM缓存。结果见`data/Planning/relaxed_fusion_v2_trainval_20261004/report.md`，关联相关21项测试（6项新增）通过。

2026-10-04 T1/S1新接口已实现：tools/object_vlm/与用户脚本run_t1_s1.sh，共用dual_object_a1_v1_20261004缓存929/157/480，A1+50m+32ego/16RSU，31项相关CPU测试（10项新接口）通过。T1对象文本、S1共享encoder+xyz/source/quality soft tokens，保留ego图像及相同state/history文本，自回归轨迹/LoRA8/8。统一context7168。真实7Bsmoke因外部GPU任务占满显存未成功，用户明确自行运行，禁止自动启动训练/等待队列；尚无T1/S1成绩。详见t1_s1_object_vlm_implementation_20261004.md与../tools/object_vlm/README.md。

2026-10-04用户四卡修订：正式训练改为TRAIN_GPUS=0,1,2,3，先T1四卡6epoch，再S1四卡6epoch，均从基础初始化，有效batch仍4；smoke仍单卡并行且两组都成功才训练。torchrun梯度同步、最后singleton不重复/丢样本，validation四卡分片/汇总、rank0保存；不自动启动GPU，用户执行脚本。此前两组正式单卡并行描述已被本修订取代。

用户随后释放0–3卡，并明确授权本轮运行脚本smoke；仅smoke，未授权本轮自动开始6epoch正式训练。ACTION=smoke现在包括双组单卡7B接口检查及顺序双组四卡9train/8val短训练，记录于work_dirs/T1_S1_objects_4gpu_smoke_20261004。正式T1/S1四卡各6epoch继续由用户启动。13项本轮接口/四进程CPU测试通过。

2026-10-04最新实测：work_dirs/T1_S1_objects_4gpu_smoke_20261004中T1/S1真实7B单卡梯度/对象敏感性/重载生成检查均通过，双组0–3四卡9train/8val、3更新、分片验证与保存均通过；34项相关CPU测试通过。正式6epoch尚未启动。短probe严格轨迹解析0/8，不能作为规划效果。正式脚本为ACTION=train TRAIN_GPUS=0,1,2,3 RUN_ROOT=work_dirs/T1_S1_objects_4gpu_smoke_20261004 bash tools/object_vlm/run_t1_s1.sh；T1先四卡6epoch，再S1四卡6epoch，仍由用户启动。结果见该目录report.md和smoke_summary.json。

2026-10-04 BEV256用户运行状态：D1完成675帧原生独立AP，car38.91%、bicycle24.64%、pedestrian3.27%、三类mAP22.27%，较BEV200下降约0.93个百分点；未微调。D2准备完成但推理前GPU0占用21834MiB被保护阈值拦截，尚无D2预测/AP。随后0–3卡占用已回落；可仅接续D2，保留D1。详见bev256_detector_experiment_20261004.md末尾与evaluation_results/D1_D2_bev256_val_20261004_220923_525810/d1/summary.json。

T1正式四卡训练曾启动（记录step10/1398），用户Ctrl+C终止，所有进程已退出；无正式epoch checkpoint，S1未启动。T1_train已由用户重命名保存。重复启动的length_audit_train目录冲突已修复于启动shell脚本：每次独立审计、错误可见、训练tee输出；只做DRY_RUN验证，没有代用户重启训练。

2026-10-05正式结果确认：T1/S1均完成四卡6epoch、1398次更新，六个epoch checkpoint与optimizer文件齐全，宿主机无残留训练进程。T1最佳epoch5，val主平均L2=0.6925203365m；S1最佳epoch6=0.6358840247m；两者最佳均157/157解析成功，4.5sFDE分别3.7897720543/3.5293594817m。单seed42验证集探索，不证明RSU增益；test未评测。产物work_dirs/T1_S1_objects_4gpu_smoke_20261004/T1_train和S1_train下complete.json/best_checkpoint.json/history.jsonl。


2026-10-05 BEV256双端完成：D1/D2各675帧推理与AP均成功，hash及同协议检查通过；三类mAP22.27%/41.13%，较BEV200下降0.93/0.20个百分点。D2 car63.45%、bicycle47.38%、pedestrian12.56%、十类mAP12.34%；自行车/行人提高约1.43/1.42个百分点，car下降约3.44个百分点。直接插值无总体收益，默认保留BEV200；未启动微调或规划评测。完整对照为evaluation_results/D1_D2_bev256_val_20261004_220923_525810/bev200_vs_bev256.md，新增combined_summary.json汇总双端，未覆盖原summary。


2026-10-05 test评测脚本已提供（尚未运行GPU）：`tools/object_vlm/eval_t1_s1.sh`和`evaluate.py`按val冻结T1 epoch5/S1 epoch6，默认0–3四卡依次生成两组完整480帧test；复用训练时期的生成/解析/指标实现，核对checkpoint、cache、base和代码hash，输出逐帧预测、指标及对比report.md。启动命令：`EVAL_GPUS=0,1,2,3 bash tools/object_vlm/eval_t1_s1.sh`。用户自行启动，不自动评测。最终CPU预检查通过，证据`evaluation_results/T1_S1_test_preflight_final_20261005/frozen_selection.json`；新增3项测试通过，覆盖coverage优先选型、拒绝未完成训练、全部解析失败时不产生虚假误差收益。没有新test成绩。

2026-10-05 UniMM-V2X检测头已核查：仍为BEVFormerTrackHead；单端stage1为1500queries、BEV200、TopK300，单端stage1检测路径MoE未开启（encoder默认False、检测decoder False）。不是现成BEVDepth/深度监督替代；未评测UniMM权重。见detector_optimization_research_20261004.md末尾。


2026-10-05 test长度超限修复：用户首轮test评测在T1单帧prefill 7180>7168时退出，尚无完整test成绩。原预检查漏审计test提示长度；7168为训练包装器guard，基础Qwen配置支持32768。evaluate.py现对完整480帧两种表示的观测提示进行CPU长度审计（不读轨迹答案），记录T1最大7180/1帧超guard、S1最大1242/0帧超guard；统一评测guard提升到8192，额外检查512生成预算在模型context内。checkpoint加载后显式覆盖model.max_length，原model.py及训练权重/代码保持不变；不截断、不删对象、不重新选epoch。修复NCCL销毁后异常hook调用get_rank的连带报错。4项评测CPU测试通过，最终预检查见evaluation_results/T1_S1_test_context_preflight_20261005/frozen_selection.json。用户仍用原命令启动，自动新输出目录，旧失败目录保留；本次未启动GPU。

2026-10-05双端透视辅助监督已实现：保留UniV2X BEV200，新增训练期FCOS投影框＋log中心深度分支，共享FPN且不改变推理解码。四组D1/D2×plain/aux配置，各6epoch、FP32、解冻layer4/FPN/检测分支、BN冻结；独立Native scene划分train1190/dev327，4帧scene0087排除。预检为work_dirs/perspective_aux_preflight_20261005，CPU真实四模型权重加载/五帧队列检查通过，新12项及原生7项测试通过。GPU smoke/训练/AP均待用户运行，禁止自动启动。入口tools/perspective/run_perspective_aux.sh，完整说明tools/perspective/README.md与docs/perspective_aux_detector_20261005.md。这是透视监督增量，不是完整BEVFormer v2，未读取点云，未声明收益。

2026-10-05最新授权：用户明确要求等0–3卡空闲后自动执行透视辅助监督smoke→train→evaluate，覆盖此前本实验不得自动等待/启动的限制。tools/perspective/wait_and_run.py已实现，5项测试通过；监控PID499662运行中，初始状态waiting/next_phase=smoke，连续3次每60秒显存≤2048MiB、利用率≤5%、旧大占用PID退出后启动。单例锁/失败停止/不覆盖产物。当前核查尚未开始GPU smoke；动态状态请读work_dirs/perspective_aux_preflight_20261005/auto_wait/status.json及monitor.log。


2026-10-05 T1/S1完整test完成：evaluation_results/T1_S1_test_20261005_114004，两个complete.json均480帧、同一冻结selection hash，均480/480合法解析。T1 val-selected epoch5：L2@1/2/3s=0.1098818/0.5764141/1.5094839m，主平均0.7319266m，FDE4.5=3.824675m；S1 val-selected epoch6：0.1029771/0.5642284/1.4801853m，主平均0.7157969m，FDE4.5=3.794841m。S1主平均下降2.2037%（绝对0.01613m），小于val的8.18%，不能表述显著或稳定优势；单seed且无ego-only对照，不能推断RSU收益。tokens3076.45→1198.425（少61.05%）；生成阶段计时1692.68s→1550.50s（少8.40%，不含加载）；峰值allocated显存21,454,255,616→17,272,263,168bytes（少19.49%）。重新由逐帧预测和原始GT计算三时域平均与保存指标一致；S1在254/480帧、12/19场景改善，其余226帧/7场景退化，paired_analysis.json已保存。本轮仅分析完成产物，无新增训练/评测。建议保留S1作为紧凑接口候选，多seed确认精度差异；若研究路侧价值需补同构ego-only，不用test调参。

2026-10-05实验表更新：融合方案3.1.1新增T1、I1/S1、I2、I3方案表与首轮test结果；优化实验计划2.2总表追加T1/I1实测、I2/I3待实验行，保留C0–C7。Final L2定义4.5sFDE，单列三时域平均；显存十进制GB allocated峰值，Latency/BPS/碰撞等未评测留空。融合方案第2节原文哈希核对未改。

2026-10-05补长时域baseline表：融合方案3.1.1和优化实验计划2.2追加用户给定V2X-VLM参考值（2.5/3.5/4.5s L2=1.09/1.12/1.42，Avg1.21；Collision%=0.02/0.03/0.03，Avg0.03），明确来源和协议待核实，不作为公平排名。已有预测重算T1三时刻L2=0.9740/2.1524/3.8247，Avg2.3170；S1=0.9564/2.1246/3.7948，Avg2.2920，碰撞未评测。4s分别2.9286/2.9043。证据long_horizon_metrics.json，无新增GPU推理。


2026-10-05长时域误差诊断：依据同480帧预测重算，horizon_error_diagnosis.json保存全九步mean/median/p90及XY分量。S1 L2@2/3/4.5s=0.5642/1.4802/3.7948m，median=0.4841/1.2357/3.2063m；467/480帧3s劣于2s，不只是少数离群。平均|dx|=0.4285/1.0460/2.5213，平均|dy|=0.2502/0.7495/2.1451，非只有纵向。回顾性按GT末段(3→4.5s)方向相对当前ego前向角绝对值>0.15rad粗分（不等同正式转弯类别/意图标签）：239帧该组S1 3s/4.5s=1.8485/4.8415，T1=1.8167/4.7549；241帧其余组S1=1.1150/2.7569，T1=1.2049/2.9021。S1改善主要在该近直行粗组，曲线/转向难题未解决。固定已有state的CV诊断2/3/4.5s=1.1899/2.9029/6.8041，CA=.8273/1.9488/4.8657；T1/S1均优于这些简单外推，不能称仅复制恒速。误差随时域非线性增长符合运动变化难度，但没有定位到具体模块的因果证据。当前监督为答案token CE，无连续坐标/曲率辅助loss；StateEncoder未接入，两组同状态文本。后续优先审计速度/加速度/历史数据实际来源、转向场景与连续几何监督消融；未新增训练/推理。


2026-10-05透视监督最新修复：首次监控已因D1/plain首个前向OOM停止（约44.97GiB），未完成更新。新增FP32激活重计算及Torch1.9/旧MMCV静态图DDP train_step桥接，保持BEV200/五帧/图像/学习率/原生split；旧目录及中间失败v3保留。当前监控PID2081278，目录work_dirs/perspective_aux_memory_v4_20261005，smoke→train→evaluate授权继续有效。四模型CPU检查及2项重计算单测通过；本机CPU Gloo尝试超时，不当作通过。D1/plain单卡3次更新、有限权重/优化器和非零共享分支梯度通过，峰值显存日志11661MiB；四卡/其他组正在验证，尚无新AP。后续动态状态读该目录auto_wait/status.json；详见docs/perspective_aux_detector_20261005.md。

2026-10-05 I2本地代码／原文核查完成：新增docs/i2_visual_spatial_queries_code_audit_20261005.md及融合方案3.1.2，核实OmniDrive本地LLaVA-Llama 7B/EVA-02/Q-Former-PETR（config未显式写Vicuna版本）、ORION Vicuna v1.5/EVA-02/QT-Former历史memory、VGGDrive Qwen2.5-VL-7B＋冻结VGGT层内CVGE。首版设计I2-A同图像/对象输入，P保留，对象条件query读SigLIP，Q替换S，状态历史文本；不可见对象几何fallback。I2-B路端视觉及VGGT层内分支另外控制新增信息/权重。当前完成核查和开发方案，尚未实现I2入口或启动GPU；用户自行并行处理状态/轨迹，本分支不改。融合方案第2节哈希核对未改。


2026-10-05 15:58修复验收：四组D1/D2×plain/aux全部完成单卡及四卡smoke，每个smoke连续3次参数更新，权重/optimizer均有限；训练分支均有非零梯度，四rank各分支梯度L1汇总逐次一致。峰值allocated日志MiB：D1/plain单11661、四11662；D1/aux单12019、四12023；D2/plain单13221、四13226；D2/aux单13580、四13586。证据work_dirs/perspective_aux_memory_v4_20261005/memory_fix_audit.json及四份smoke_complete.json。12项辅助测试、2项重计算测试、5项等待测试通过，另1项CPU Gloo集成显式跳过；真实GPU检查作为DDP验收。监控PID2081278已自动进入train，先D1/plain四卡6epoch，随后D1/aux、D2/plain、D2/aux，再evaluate选型。正式训练尚未完成，尚无新AP；读取该目录auto_wait/status.json获取实时状态。wait_and_run.py后续启动若失败会在报错中指出阶段日志及最近worker日志，并避免重复打印traceback。

2026-10-05 I2开发前本地Git快照：提交范围为T1/S1对象VLM、test入口、关联/补偿、依赖continuous接口、测试与研究文档；外部参考仓库及并行检测改动保留工作区，模型/数据/日志不入Git。本次48项相关CPU测试通过，分布式专项已有此前四卡smoke证据；脚本bash语法与暂存文件检查通过。
