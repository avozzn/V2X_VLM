# 下一轮计划：保留图像 tokens，学习空间对齐，显式状态条件化，VAE＋GRU 规划

> 最新用户修订：先比较I1/I2/I3三种VLM输入接口；所有RSU信息必须经过VLM，禁止任何planner侧RSU旁路。当前权威实现顺序见[融合方案第1～3节](roadside_agent_token_fusion_plan.md)。本文下方旧阶段/图示若冲突，按该修订执行；VAE延后，自车state旁路不纳入三组主比较。

日期：2026-10-04。本文按用户本轮要求记录接续调研与开发计划；不启动训练、不修改模型代码。下一轮先读取本文件、PROJECT_CONTEXT、F0审查和轨迹头文献研究报告，再核对实时产物。

## 1. 用户已明确的方向与本次范围

1. 保留当前原始图像经LLaVA视觉编码器得到的image tokens，不采用AURORA完全以融合semantic tokens替换视觉输入的接口。
2. 借鉴AURORA的spatial queries：对象/地图表示在ego坐标系对齐、关联、融合；探索有监督的路端→车端空间/时间适配，参考UniV2X代码。
3. 引入专用规划token/query，读取其hidden连接连续decoder。
4. 借鉴条件VAE＋GRU轨迹生成与训练逻辑，同时验证数值自车状态显式输入；形成与AURORA不同的实现路线。
5. 保留E2 LoRA冻结特征＋相同MLP的先行对照，先判断驾驶适配和readout问题。

这是已确认的研究方向，不代表所有参数、数据来源、loss权重和开发接口已经确定。本轮只写计划；下轮用户开始开发后按阶段执行。不要仅因读到本计划就自动占用GPU、跑全量实验或删除旧产物。

“保留图像”首版指保留当前F0车端图像；路端首版走结构化消息，不默认新增路端图片传输。若后续实验使用路端图像，需要单列输入/通信设置。车端GT文本的保留与空间tokens替代试验分别记录，不能在加入新模块时无记录删除旧输入。

## 2. 已知事实、诊断和证据路径

| 项目 | 已核实结果/边界 |
|---|---|
| 旧正式F0 | 929train/157val，100epoch，best epoch78；主L2=3.9697m，4.5s FDE=9.8169m；工程完成、验收未过 |
| 恒速基线 | 同val主L2=1.4660m |
| 数值state ridge探针 | vx/vy/ax/ay直接输入，train-only标准化与拟合；val主L2=0.6312m，4.5s FDE=3.4354m |
| 冻结hidden ridge探针 | 同val主L2约3.33–4.06m；不证明所有readout或非线性decoder无效 |
| 当前F0权重 | 基础7B LLaVA，未加载E2驾驶adapter；缓存detach，只训练head |
| E2权重 | checkpoint-932有LoRA adapter；其旧输入与当前F0不同，旧E2成绩不能代替新同输入实验 |
| 状态证据边界 | 0.6312m是诊断，不是正式baseline/test成绩；原始状态时间因果来源待专项审计 |
| 当前消息约定 | 独立车/路来源；RSU velocity-disabled、age未编码/未补偿；约2.2s陈旧风险需复核，不能未经审计启用未来推导速度 |

长期证据：

- `docs/f0_training_audit_20261004.md`。
- `docs/trajectory_head_training_research_20261004.md`，含AURORA机制及E2对照补充。
- `work_dirs/continuous_F0_ego_gt_seed42/audit_20261004/{cause_diagnostics,state_probes}.json`。
- `checkpoints/meta_v2_physics_seed42_4gpu/checkpoint-932/{adapter_config.json,adapter_model.safetensors}`。
- `tools/continuous/{prepare,data,cache_features,model,train,predict}.py`及README。

## 3. 目标架构：三条互补信息通路

```text
车端原始图像 → 现有视觉编码器/投影 → 原始 image tokens ──────────┐
                                                            │
独立车端对象、路端对象(/后续地图)                              │
  → 标定坐标变换 → 学习时间/残差适配 → 对象关联与融合             │
  → geometry-aware spatial tokens → 语言空间投影 ───────────────┤
                                                            ├→ VLM＋LoRA
观测文本/固定规划提示＋数值state tokens(待验证) ─────────────────┘
                                                               ↓
                                                          <wp> hidden
                                                               ↓
数值state/因果history → StateEncoder → h_state ───────────────→ 条件 c
                                                               ↓
                                             条件VAE prior＋GRU decoder
                                                               ↓
                                          运动学参考轨迹＋场景条件残差
                                                               ↓
                                                         9×2 waypoint
```

首版明确保证数值state的decoder旁路；向VLM增加state tokens为后续可控分支，以免连续数值仍被语言压缩。主要条件可定义为c=Fuse(h_wp,h_state)，是否让GRU再次cross-attend空间tokens用消融决定，不一开始增加两套重复融合。

新增spatial tokens是原始image tokens之外的额外输入，两者必须通过实际token长度、mask、梯度与干预测试确认可用。“图像仍保留”不能仅看数据JSON里有image字段。

## 4. 相对AURORA的区别与候选研究问题

| 维度 | AURORA原文明示 | 本项目目标 |
|---|---|---|
| 视觉接口 | 融合semantic tokens替换视觉占位 | 保留常规image tokens，同时增加结构化spatial tokens |
| 时空适配 | CQAF标定对齐、特征融合 | 标定为几何基底，额外研究有监督时间/残差/特征适配 |
| 数值state | 当前核查未完整明确其独立状态decoder通路 | 显式state encoder和decoder条件；可研究运动学参考＋残差 |
| planner | waypoint hidden→条件VAE/GRU；有MLP替代 | 同时比较状态MLP、确定性GRU、条件VAE/GRU，防止把结构复杂度当收益 |
| 数据与评测 | V2XBench闭环、QA监督 | 当前DAIR子集开放环先验收；不声称已复现闭环安全收益 |

候选问题：在保留原始图像表示的条件下，结构化时空消息是否提供额外协同价值？显式数值状态是否改善可泛化规划接口？场景条件VAE能否建模超出运动学外推的行为残差？这些是待验证的研究问题，不是新颖性/效果已经成立。

## 5. 下一轮先做的代码与数据调研

### 5.1 UniV2X时空模块的真实边界

已读本地`UniV2X/projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py`：

- `AgentQueryFusion`对路端reference points应用标定矩阵逆/转置约定，进行匹配、query融合及未匹配query补充。
- 特征对齐层把query特征与flatten后的3×3旋转矩阵拼接，使用可学习Linear转换。
- 该文件未定义独立的`L_align`返回，也没有完整的跨消息时间预测机制。不能把“可学习对齐层”解释为论文已有我们要求的专门转换loss。
- `lane_fusion.py`有类似空间feature/position对齐。
- `detectors/univ2x_track.py`有速度×time_delta及坐标变换，但这是跟踪时序路径；是否用于实际跨路端消息延迟需沿调用链核查。
- 本地检索未定位完整FlowQueryNet独立实现；下一轮对照UniV2X原文、官方repo版本及调用配置确认，不把tracker时序更新直接冒充FlowQueryNet。

读取入口：
`/home/zzn/V2X_VLM/UniV2X/projects/mmdet3d_plugin/univ2x/{fusion_modules/agent_fusion.py,fusion_modules/lane_fusion.py,detectors/univ2x_track.py,detectors/multi_agent.py}`。

产出：一份“官方描述→本地函数→实际调用→输入/梯度/loss→可复用程度”的接口表，标注版本与未实现部分。

### 5.2 输入合同与监督可得性

核查train/val/test token和scene、图像路径、原生车/路GT来源、坐标/时间、ego-state与history。查明跨端track ID是否可关联；不能假设原生ID跨端相同。监督标签允许使用GT建立训练对应关系，但推理输入必须来自各端实际可得观测；训练构建过程不能将cooperative GT整体当成独立输入。

重点：速度/加速度是否由未来轨迹差分得到，历史是否真正截至当前时刻；route/map可用性；RSU消息时间是否≤ego时间；pose at t_r / t_e的组成；传感器坐标和车辆坐标区别；yaw与尺寸顺序。

地图为可选后续输入：当前首版未接独立车/路map decoder，先验证object spatial tokens。没有可用地图表示时，不放占位假地图、不加入空的boundary loss。不得默认把UniV2X所有map heads已训练可用。

产出：新版数据schema、来源/因果审计表、匹配有效数量及age分布。此检查是后续空间监督和state基线的前置条件。

## 6. 分阶段开发与实验

| 阶段 | 工作与产物 | 验收与下一步 |
|---|---|---|
| P0 | 数据/坐标/时间/state审计；UniV2X/AURORA接口核查 | 来源、因果性、调用链清楚；缺失接口有具体降级方案 |
| P1 | 基础VLM vs E2 LoRA冻结特征＋相同MLP；正式state-only baseline | 仅权重变化的A/B可复现；state基线有合法来源和完整val |
| P2 | state encoder＋MLP/确定性GRU；直接轨迹 vs运动学残差；readout对照 | 在相同预算下验证状态通路和readout，不宣称VAE收益 |
| P3 | 保留image tokens并加<wp>；在线head/token预热→LoRA联合训练 | 无未来泄漏，image/spatial/state梯度及读出位置正确，checkpoint重载一致 |
| P4 | 先固定几何，再训练空间/时间适配与spatial token投影 | 模块独立几何误差改善、空消息正确、遮挡对象不被筛掉 |
| P5 | 条件VAE＋GRU与确定性GRU对照；逐步联合适配 | 推理只用prior，KL/latent可用性、完整val及采样协议稳定 |
| P6 | 同构新版F0/F1/F2及模态、对齐、状态消融 | 能归因协同收益；冻结选型后统一test报告 |

P3/P4可以在代码设计上衔接，但实验上须保留不带RSU的F0和固定几何版本。P5不必等整个地图系统完成；先验证ego-only的状态条件VAE。所有升级使用新实验版本，旧M1结果保留为历史对照。

### P1：先排除表示权重差异

A=基础LLaVA冻结特征＋旧head；B=基础LLaVA＋E2 adapter冻结特征＋同head。固定当前F0观测、prompt、readout、head初始化和预算。重新提取B缓存，保存adapter身份/哈希，不能复用A缓存。

审计E2训练及validation选型与当前split关系。由于E2在旧感知协议上训练，即使B失败也不能简单归因“驾驶LoRA无效”；记录迁移差异。C=数值state-only，报告train/val和恒速/恒加速度基线；CA只是外推基线，不默认更优。

### P2：显式利用数值state

首版state建议为当前ego-frame vx/vy/ax/ay＋mask-aware因果history；缺失字段用mask，不伪造有效值。统计量仅train拟合，固定物理缩放也可以，但必须随checkpoint保存。

比较四组：state-only；hidden-only；hidden＋state；hidden＋state预测运动学残差。残差参考优先CV，CA作为另一个对照，不机械外推4.5s加速度。残差target=未来GT−合法参考轨迹，未来GT仅用于loss。

state编码既可以拼接至decoder条件，也可构建数值state tokens给VLM；先做前者保证数值可读，后者单独比较。不要同时改变readout、state、decoder和loss后将改善归于单一组件。

### P3：保留image tokens的专用规划接口

新增真正的<wp> token，定位在观测信息/新增spatial tokens之后、任何future GT答案之前；这是本项目因果接口设计，不称为AURORA原始模板。

调研LLaVA 4.45.2的图像展开与language_model输入路径，沿现有image merge后追加/插入spatial embeddings，保持attention mask、position IDs和token位置映射正确。首版不依赖脆弱hook猜最后一个token。不能只用inputs_embeds绕过视觉合并导致图片实际丢失。

阶段A冻结基座，训练head、投影及任务embedding；只更新新token行或独立任务embedding，避免为一行token无意解冻整个词表。阶段B在线训练语言LoRA和head，移除阻断所需梯度的no_grad；视觉encoder保持冻结。若空间投影/任务embedding在冻结语言基座之前，仍必须通过语言计算图回传，不能给整段language forward加no_grad。

E2 LoRA作为初始化候选，基础VLM新LoRA作为对照。QA混合需要有可用数据与明确来源，先不编造任务。save/load包含adapter、head、task embedding、spatial/state encoder、tokenizer与归一化合同。

### P4：路端→车端的可学习时空适配与监督

几何基底使用已知标定/pose，不让网络从零猜刚体坐标变换。定义：路端观测在t_r，ego规划在t_e，age=t_e−t_r≥0；先明确两时刻坐标变换，再预测动态对象在当前时刻的位置/特征。

候选模块输入：路端geometry/query、有效mask、age、相对pose/rotation、合法历史(若可得)、uncertainty；输出：ego-frame对象token、可选位置/速度残差、关联置信度。首版13D几何编码与真正detector latent query分开命名，不能把GT MLP token称为已训练视觉detector query。

候选监督（均为本项目设计，不声称原论文原样）：

- `L_geo/time`：对可靠匹配对象，比较预测ego-frame当前center与train时刻GT；动态残差必须在坐标/时间一致的标签下训练。
- `L_yaw`：在需要时用sin/cos或周期角误差，避免±π跳变；尺寸/类别监督按可得性启用。
- `L_feat/match`：有合法对应关系时，用共同投影空间的匹配/对比监督；不同backbone hidden不能默认逐维相等，teacher一侧stop-gradient并加入防坍缩约束。
- `L_plan`：最终规划监督，检验空间适配是否对轨迹有益。

在输入已经是精确当前GT且标定已知时，再对同一解析变换输出计算L_geo只是平凡恒等任务，不能据此宣称学会跨域转换。真正可学习目标应来自真实异步消息、detector误差或明确标注的train-only扰动；合成延迟/扰动与真实实验分表报告。

先比较解析标定、标定＋age、标定＋监督时间补偿、标定＋学习feature适配。首版velocity-disabled保持为对照；启用路端速度前核验来源，只允许截至消息时刻可得速度/历史。ego-state仅描述自车运动，不能替代路端对象各自的运动。不能用一辆自车的加速度补偿所有路端目标。

匹配loss只在可信共同可见对象上计算；路端独有的遮挡对象仍保留给planner，不因没有车端匹配而删除。地图静态元素与动态对象分开处理，不能都按velocity×age推进。缺失GT/current对应或track ID不可靠时，记录监督mask与coverage，不强制错配。

### P5：条件VAE＋GRU，显式state与运动学残差

下述是标准条件VAE的拟议实现，需要下一轮对照AURORA源码/附录及现有库核查，不冒充其完整公开实现。

条件c=Fuse(h_wp,h_state)。prior `p(z|c)`只看当前观测；training posterior `q(z|c,Y_gt)`可以编码future标签。训练用posterior重参数采样，GRU按0.5s展开9步displacement/残差；推理只用prior，累加位移并可加CV参考轨迹得到waypoints。

目标先从 `L_waypoint + beta*KL(q||p)` 起步；KL系数/annealing先做小规模对照，监测posterior collapse、先验推理与后验训练gap。可选velocity/displacement辅助loss需独立消融，不加入没有有效checker/地图标签的collision/boundary项。

推荐比较：同条件MLP、确定性GRU、条件VAE＋GRU；VAE分别比较直接轨迹与运动学残差。相同state条件、数据、预算，避免把状态增益算成VAE增益。

固定正式推理策略后再评测：例如prior均值单轨迹与固定seed采样的两套协议。prior均值不保证在非线性decoder下等于平均轨迹；多样本要有不使用GT的部署选择规则。best-of-K/minADE只能作为明确标注的诊断，不能替代可部署主L2。teacher forcing若使用需审计训练/推理差距，正式推理不输入真实前一步future waypoint。

## 7. 整体loss与训练阶段边界

拟议联合目标：`L_total = L_waypoint + beta L_KL + lambda_geo L_geo/time + lambda_match L_match + lambda_qa L_QA`。每项按样本/对象有效mask归一化，非该任务样本不强制计算；权重和单位必须调研后确定。

顺序：先训练state/readout基线；独立验证aligner；预热新增投影/head；再开启语言LoRA与planner的联合训练，最后才有限联合aligner。感知D1/D2先冻结；不要为了新增规划任务重新跑整个检测预训练。

训练与val按scene隔离。validation用于选型；test仅最终冻结版本报告。除既有内部split外，后续官方split复现独立列预算，不混用数字。

## 8. 必要消融与验收条件

| 对照 | 要验证的贡献 |
|---|---|
| state-only vs相同state＋图像＋对象 | 场景信息是否超出运动学预测 |
| 原始image tokens有/无 | 保留图像是否实际改善；分别训练对应组 |
| 图像错配/对象错配/状态错配 | 固定权重敏感性诊断，不能单独代替训练消融 |
| 基础VLM vs E2 LoRA vs在线LoRA | 表示驾驶适配与规划联合监督 |
| 几何固定 vs可学习时间/feature适配 | 专门转换监督是否带来独立收益 |
| 对齐loss有/无，age分桶 | 监督价值、消息陈旧与置信度校准 |
| 同条件MLP vs GRU vs VAE＋GRU | decoder/概率latent的独立贡献 |
| VAE有/无state、直接预测/残差预测 | 是否只靠state shortcut；残差先验贡献 |
| 新版F0 vs新版F1 vs新版F2 | 同构、同预算下的路侧价值与detector误差 |

工程验收：坐标旋转平移往返、真实时间pose合成、空/全padding对象、遮挡独有对象、缺失速度、age符号、图像token保留、<wp>定位、batch>1、loss mask、梯度覆盖与prior推理无GT、save/load一致。只写验证真实风险的测试，不为文档改动添加测试。

正式指标：L2@1/2/3s及三者平均、全9步mean、3s和4.5s FDE分别记录、coverage/missing/invalid；对齐额外记录center/yaw误差、匹配质量与监督覆盖、age分桶。完整157条val，与恒速及正式state-only比较；不能以极小smoke宣布通过。多seed优先至少3个，额外预算单列；必要时按scene bootstrap报告不确定性，7个val scene下局限仍要说明。

## 9. 实施入口、产物与接续清单

保持已有`tools/continuous/`旧MLP链路可复现；新分支建议`tools/continuous_v3/`或明确backend配置，下轮按复用程度决定。候选新文件：state_encoder、spatial_aligner、token_adapter、online_readout、conditional_vae_gru及独立train/eval入口；不是本次已创建的代码。

新数据版本建议`continuous_v3_image_spatial_state`；新产物前缀建议`work_dirs/continuous_v3_*`，具体名称/超参按实际run.json落盘。Oracle geometry、Detector geometry和真实latent query分开cache版本；保存tokenizer/base/adapter身份、来源、timestamp、prompt、split、归一化、loss、推理采样策略与seed。

下一次开始时按此顺序接手：

- [ ] 读取本计划和当前上下文，确认有没有更晚用户修订/训练产物。
- [ ] 完成P0状态/时间/坐标审计，核查UniV2X的跨端时空实际调用和FlowQueryNet可用性。
- [ ] 写出P1的A/B/C实验配置、cache身份与统一val协议。
- [ ] 先开发E2 adapter特征提取与state-only基线，做save/load与小样本工程检查。
- [ ] 核对GPU与既有进程，得到下一轮开发/运行指令后再执行正式实验。
- [ ] 根据P1/P2结果实现保留图像的<wp>在线通路，不提前把VAE和所有loss一次加满。
- [ ] 逐步开发aligner和条件VAE＋GRU，更新实验矩阵、验收与实际结果。

当前完成：计划落盘、关键本地接口定位、已知事实/待核查分离。未完成：P0专项审计、任何新模型实现、任何新训练与正式消融。

## 10. 原文与研究边界

[AURORA §4.1–4.3及Appendix B](https://arxiv.org/html/2608.21032v1)：spatial/semantic表示区分、waypoint接口、语言LoRA与VAE/MLP/diffusion训练；原文未完整给出所有模板/采样实现，不能补成事实。

[UniV2X官方仓库](https://github.com/AIR-THU/UniV2X)：下一轮与本地版本核对；本轮确认的是本地agent_fusion.py/lane_fusion.py结构，未宣称独立alignment loss或FlowQueryNet已复现。

本计划由AI辅助整理，依据用户已选方向、既有审查产物和原文；目标是便于接续，非系统性文献回顾或已完成的实验报告。保留图像tokens＋有监督时空适配＋数值state条件VAE是待验证组合；与AURORA不同不自动意味着学术新颖性，下一轮另查相关工作与代码。

## 11. OmniDrive本地视觉token接口核查与StateEncoder解释

本次读取用户指定的`/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main`源码，未复现实验，不把该目录所有配置当成官方主实验。

StateEncoder是拟议的小型可训练网络：把vx/vy/ax/ay及截至当前的history/mask编码为h_state，与VLM的h_wp一起组成decoder条件c；它不是另一套VLM，也不是读取未来。当前标量可用MLP，history可用mask-aware pooling/GRU等，具体结构待实验。条件VAE prior p(z|c)提供轨迹latent，GRU按时间递推位移；training posterior可读取未来GT标签，推理只能用prior。state旁路用于避免所有数值都经过文字和单hidden读取，并不代表只靠state或性能一定提高。

本地OmniDrive已核查：

- `projects/mmdet3d_plugin/models/detectors/petr3d.py:278–294`：检测/map head输出的det_query/map_query拼接为vision_embeded，传给lm_head的images参数。这里images实际是embedding，不是RGB。
- `models/dense_heads/streampetr_head.py:500–581`：图像feature map展平为memory；附加learned queries经带3D/时序信息的Transformer得到vlm_memory；它与用于预测bbox的queries在输出处理中分开。不是仅把最终检测框转成token。
- 同文件输出投影到语言维度，并将当前command/can_bus、memory_canbus及历史ego pose编码成一个can_bus_embed token拼接到vlm_memory。说明数值状态token可与图像特征并存；它是语言输入状态token，不能直接当作已有VAE/GRU decoder旁路。
- `models/dense_heads/llava_arch.py:64–123`：已生成的image_features在<image>占位处插入文本embedding序列；没有在该路径重新读取RGB或调用常规CLIP patch encoder。
- 已读示例config `projects/configs/OmniDrive/mask_eva_lane_det_vlm.py`：检测与地图均num_extra=256，out_dims=4096；StreamPETR有条件state token拼接。token预算与维度应按实际配置/运行tensor核验，不硬编码照抄。

可借鉴：learned query从视觉memory读取并压缩信息、3D位置和时序编码、projector对齐语言维度、数值state token注入、模态embedding与mask合同。注意用户要保留现有LLaVA image tokens，因此首选“常规image tokens＋额外query/spatial tokens”，而非照搬OmniDrive以query输出替代常规视觉输入的接口。后者可单独作为压缩消融，需另确认是否仍符合用户希望保留的图像信息形式。OmniDrive的3D位置构建依赖其相机标定/多视角feature和训练，不把新增随机query当作已有3D理解能力。

下一轮补查：position_embeding具体深度/标定构造；PETRTemporalTransformer注意力mask；petr_head_map的query路径；本地配置来源与checkpoint能否迁移；LLaVA 4.45.2额外token与原始image merge的共存接口。暂不更改原架构选择或启动训练。

## 12. 视觉接口方案评估：不要把三种机制当成互斥选择

本轮用户询问是否仍需保留常规image tokens；这是重新评估，不是已经授权删除该分支。本节是建议，前述保留图像的用户选择在对照结果/新决定前继续有效。

补充原文：[OmniDrive v2](https://arxiv.org/html/2405.01533v2)，§3.1/3.2/3.3与§4。它包含Omni-Q和Omni-L两条路线：Omni-Q把carrier queries与perception queries交互，再从多视图feature读取信息；carrier输出进入LLM。Omni-L保留flatten图像特征，经MLP对齐，加入3D位置编码(初始权重为零)。本地上一节读到的PETR/query代码仅支持所检查路径，不代表OmniDrive所有版本都是query-only。

三者分别是：常规image tokens的视觉接口；Omni-Q的可训练视觉压缩/语言对齐接口；AURORA的跨端空间融合及semantic接口。三者都以图像特征为上游，query-only也不等于无视觉。AURORA空间query与语义token是不同层级，13D几何MLP也不能冒充视觉carrier query。

当前推荐：短期保留已有LLaVA视觉路径，先验证state/readout与驾驶适配；长期以Omni-Q启发的视觉carrier＋AURORA/UniV2X启发的空间融合为候选主线，只有经过对照证明原始patch无额外价值才删掉patch分支。不默认“全部叠加”就是最终最优结构，也不从不同论文数据/训练预算推断谁在本项目最好。

### 12.1 三条路线的代价与风险

- Patch接口保留较直接的视觉特征和现有语言对齐权重，但长序列增加语言侧计算，显式几何较弱；特定信号/纹理是否有用需要验证。
- Omni-Q/carrier接口用有限query读取视觉feature，能结合空间/时序prior，减少进入LLM的token数；压缩可能损失信息，随机query在929条数据上从零学习存在明显训练风险。不能把新增query当成已获3D能力。
- AURORA-style空间融合直接针对车路坐标/对象关联和遮挡补充；依赖感知/map表示、标定、时间和语义投影训练。当前map/query/checkpoint尚未全部审计可用，不能以空模块宣称复现。
- 保留patch＋query能提供互补通路，也可能冗余、加长上下文或过拟合；只有训练消融与效率测量能决定保留价值。

### 12.2 分开选择视觉接口与路端融合

先固定同一个空间消息来源/融合策略、state、<wp>和确定性head，比较：

| 组 | 常规patch P | 视觉carrier Q | 同一空间输入 S | 用途 |
|---|---|---|---|---|
| V0 | 无 | 无 | 无 | state-only参考 |
| V1 | 有 | 无 | 有 | 现有视觉接口＋空间增强 |
| V2 | 无 | 有 | 有 | query能否替代patch |
| V3 | 有 | 有 | 有 | patch与query是否互补 |

S采用同一来源、合法ego/RSU geometry及同一token投影；没有地图时只比较object，不用不同GT/detector来源制造不公平差距。Q优先只用ego图像，RSU独有信息仅由固定S进入，使V1/V2/V3路端信息一致。若Q自行读取了RSU图像、S又不含同等信息，需独立标注为新增输入组。

另外把S移除的对应组作为消融，以免state/S掩盖视觉贡献。这张表研究的是当前Oracle setting下的视觉接口价值，不等价于从零复现两篇论文；后续车端detector setting需重复主要对照。

随后固定选出的视觉接口，在同一物理对象/来源下比较：无RSU、解析标定后token拼接、可学习query融合、带显式时空监督的融合。区别于第一轮视觉接口，不把CQAF引入和patch删除同时发生后的差距归因于单一机制。

### 12.3 实验预算与判据

先完成已有P1/P2，固定数值state与head；视觉接口选择阶段不同时把MLP换VAE，以免decoder贡献混淆。Q须有可比预训练/适配说明；无可用权重时，先小规模空间/语言对齐验证，不急于正式训练全部V组。相同任务loss、split、输入来源与基础LLM权重，记录新增参数、token数量、显存、时延、训练预算与预训练差异。初筛后只给候选组追加多seed，test留给最终冻结版本。

若V2在主L2和长时域、不同scene/转弯/减速样本及可用约束上与V3接近、成本更低，可选query-only；“接近”的容忍差需实验前定义，不能结果出来后改。若V3稳定优于V2且成本可接受，保留两条视觉通路。若Q尚未训练充分而V2很差，只能否定当前实现，不能否定query范式。

若各视觉组与state-only接近，先检查状态shortcut、观测/标签、接口梯度和数据场景覆盖，不依据低L2直接删掉视觉。固定权重移除patch是敏感性诊断；最终删除patch的结论应来自分别训练的V2/V3。没有碰撞checker时不比较零碰撞；开放环L2不作为闭环安全结论。

阶段结论：需要实验，但先用固定确定性decoder完成有限视觉接口消融，再训练VAE；目前没有删除image tokens、迁移OmniDrive权重或启动新实验。

## 13. 最新接口协议：先VLM输入，后轨迹decoder

用户要求三条输入路线都尝试；按roadside_agent_token_fusion_plan.md第3节执行I1/I2/I3及各自F0/F1/F2。停止将RSU encoder输出拼在h_wp之后或供planner cross-attention；I3的RSU影响通过融合semantic tokens进入VLM。先固定deterministic head比较接口，之后VAE＋GRU。自车state先作为VLM输入token，decoder旁路另作待定消融。旧P1/E2特征实验仅可作无RSU诊断，不成为新版接口准入强制前置。

## 14. 紧急修复已落地

新增I1在线train/predict，RSU soft embeddings在语言模型前合并；head只有规划hidden，旧RSU planner旁路禁用。CPU 15项测试通过，真实7B未运行。当前state/车端对象仍走文本，planning soft embedding不注册tokenizer；仅输入模块/head训练，语言LoRA更新及完整I2/I3/VAE未实现。参见tools/continuous/README.md，后续先做真实7B小规模前向/梯度smoke，再正式训练。
