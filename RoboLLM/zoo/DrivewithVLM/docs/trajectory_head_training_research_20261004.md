# 轨迹头训练策略：参考论文核查与 F0 改进建议

日期：2026-10-04。范围：`docs/references/` 的全部25个PDF文件，先做全文可解析性与关键词筛查，再深入对照与轨迹训练相关的章节、附录及部分官方来源。23个文件可提取文本；2个PDF结构无法正常解析，其中Drive-R1通过AAAI官方原文补查，编号39暂不作为证据。不是声称逐字精读全部25份。

本文回答：现有论文是否支持“冻结VLM，读一个hidden，再训练连续轨迹头”？哪些组件实际训练，训练监督怎样进入轨迹表示，哪些策略可以迁移到当前F0？

## 1. 结论与证据边界

**有论文讨论连续轨迹头及分阶段训练，但没有在本次核查中找到与当前F0完全相同、且证明有效的训练配方。**

最直接的VLM→连续轨迹头证据是AURORA：专用`<wp>`token、语言侧LoRA适配、轨迹decoder和规划监督联合训练。OmniV2X支持冻结通用视觉编码器、训练独立planner，但其编码器是DINOv3视觉模型，不是冻结整个对话VLM；planner先在大规模单车数据上预训练，然后迁移到协同数据。

因此，当前“真实7B LLaVA冻结→缓存聊天模板末token→929条样本从零训练MLP”可作为工程基线，却不能称为复现AURORA或OmniV2X的规划训练策略。论文证据说明存在值得修补的设计差距，**不能单凭差距确定F0失败的唯一根因，也不能保证加入LoRA就一定成功。**

| 方法 | 轨迹表示/接口 | 训练策略原文 | 对当前F0的意义 |
|---|---|---|---|
| AURORA | 专用waypoint token的最终hidden→VAE/GRU；另比较MLP、diffusion | 感知预训练→视觉语言对齐→LoRA+planner训练→QA/规划混合微调 | 最直接支持任务token与语言侧规划适配 |
| OmniV2X | 冻结DINOv3 patch tokens、多模态条件序列→轨迹Transformer；step-wise displacement flow matching | 大规模单车规划预训练→协同适配；视觉空间保持冻结 | 支持冻结视觉，但必须区分已有planner先验和随机MLP |
| UniAD | ego motion query与command、BEV交互→waypoint | 感知6epoch→全任务端到端20epoch | 支持预训练、专门规划query；不是VLM配方 |
| UniMM-V2X | 多层融合与运动/规划任务 | 感知40epoch→端到端20epoch，多任务损失 | 支持分阶段；不是只训练随机head |
| DiffusionDrive | 多模态轨迹anchor，BEV/agent条件→去噪decoder | anchor分类+正样本轨迹重建；训练集构建anchor | 支持空间条件与多模态监督，不能只把MLP换成扩散就称复现 |
| SparseDriveV2 | 场景特征与独立ego-state编码，路径/速度候选及评分 | 因基准不同使用不同训练阶段 | 支持独立状态通道，但不是当前连续回归头的相同目标 |
| DriveVLM | VLM推理/轨迹→Dual系统的快速planner精化 | 驾驶任务微调与额外数据共同训练 | 支持驾驶适配，不能当冻结hidden-MLP证据 |
| Drive-R1 | 文本生成CoT/动作/轨迹 | 领域SFT→推理SFT→GRPO | 是生成式训练路线，不是独立连续头的直接依据 |

页码下文均为**PDF文件物理页码，从1开始**，不一定等于论文页脚。论文自己的数值仅说明其基准上的结果，不直接移植到本项目929/157/480子集。

## 2. AURORA：最接近“VLM接连续轨迹头”的证据

原文：[Roadside-Cooperative Autonomous Driving](https://arxiv.org/abs/2608.21032)。本地：[PDF](references/arxiv/03_2608.21032_AURORA.pdf)，§4.3/4.4在第5页，Appendix B在第14–15页。

### 2.1 接口是任务token，而非任意聊天末token

论文将`<wp>`加入tokenizer，读取该token的最终层hidden，条件化概率VAE planner；GRU递归生成轨迹，并比较其他decoder。这个token承担规划接口职责，与当前assistant起始模板后的换行token并不等价。

不能推出“只加一个普通字符串就有规划能力”。论文明确的是任务token接口、语言LoRA适配与规划监督共同使用；新增embedding的具体优化/冻结细节仍需实现核查。当前prompt里的`[PLAN]`也不能未经检查就视为专用单token。

论文没有充分交代可直接照搬的所有序列位置与标签遮罩细节。本项目实现时应确保规划readout只看到当前观测/因果历史；若混合文本监督，把future GT答案放在readout之后。不能用teacher-forced未来答案之后的hidden训练头、推理时又只给观测。

### 2.2 分阶段训练确实训练了语言侧

| 阶段 | 主要任务 | 与当前问题相关的参数策略 |
|---|---|---|
| 1 | 车端感知预训练 | 建立检测/地图表示 |
| 2 | 路端感知预训练 | 建立独立路端表示 |
| 3 | 跨视角融合、视觉语言对齐 | 以VQA进行语言侧LoRA适配，尚未进入完整规划训练 |
| 4 | 激活轨迹planner | 图像backbone冻结；语言模型继续通过LoRA训练；规划decoder训练 |
| 5 | 混合QA与规划微调 | 同时维持语言能力与轨迹规划能力 |

附录明确阶段4目标为语言损失与规划损失之和。阶段5中，包含waypoint token的样本才监督规划接口；QA-only样本只贡献语言损失。MLP/VAE规划目标包含轨迹回归、车道边界与碰撞项，VAE还有KL；diffusion对应分类和去噪轨迹回归等目标。因此论文比较MLP并不等于验证“冻结未适配语言模型+只用Smooth-L1”的MLP。

**冻结策略存在原文内部不一致：**§4.4将阶段5描述为解除全部组件冻结，而附录Table 9对阶段5仍标记Freeze Backbone=True。可以确定的是规划阶段包含LoRA与planner适配；不能据此声称7B语言基座所有权重都解冻，也不应忽略正文/附录冲突。

Table 9阶段4：batch8、6epoch、base LR8e-5；阶段5：batch6、4epoch、8e-5。共同使用AdamW、weight decay1e-5、cosine调度和500步warmup。超参数用于理解训练预算，**不是推荐原样复制到本项目**。

### 2.3 可迁移的核心

可以借鉴“任务readout+head预热+语言LoRA和head联合适配”的原则。QA对齐需要真实可用的驾驶监督；现有车端GT文本是输入，不自动等于已经有了配套QA训练集。我们目前还缺少AURORA的感知query预训练、VQA数据及闭环环境，不能称为完整复现。

## 3. OmniV2X：冻结视觉编码器不等于冻结整个VLM

原文：[OmniV2X](https://arxiv.org/abs/2606.21165)，[官方代码](https://github.com/JuntongPeng/OmniV2X)。本地：[PDF](references/arxiv/17_OmniV2X_DAIR-V2X-Seq.pdf)，§III在第3–5页，预训练消融Table VI在第6页。

### 3.1 训练对象与条件结构

冻结的是DINOv3 ViT-B/16，图像patch经可训练线性投影；导航命令、可选地图与13D路端对象分别编码，加入模态标识/归一化形成条件token序列。轨迹Transformer通过cross-attention读取条件，而非将所有输入压到通用对话最后一个token。

论文总参数130M、可训练44M。冻结视觉特征可以预提取，但下游planner与投影仍学习驾驶任务，不能只看到“frozen”就归类为当前F0。

### 3.2 先有单车驾驶先验，再做协同适配

先在nuPlan的navtrain大规模单车驾驶数据上预训练，再在DAIR-V2X-Seq加入协同条件进行迁移。Table VI明确比较from scratch与预训练后不同目标数据比例，说明论文自己也把规划预训练视为关键因素。

本地已有`/home/zzn/V2X_VLM/OmniV2X-main/`。代码中可见多种freeze mode与stage开关；这些是实现选项，不能不检查具体运行配置就把所有开关默认值当成论文正式实验配方。

### 3.3 displacement是生成和监督变量，不只是最后cumsum

论文在step-wise displacement空间做flow matching，以MSE监督噪声到目标的向量场，再积分生成位移、累加为waypoint。其特定设置下，直接waypoint生成的最佳L2比displacement方案差57%。

当前F0虽然输出displacement，却在cumsum后的waypoint空间用Smooth-L1监督。因此两者**不是相同训练目标**；不能说当前已经复现OmniV2X的displacement训练，也不能断言增加位移loss就会获得57%的收益。

### 3.4 自车状态的关键反例

§III-A.2明确排除当前速度、油门和转向等测量状态，动机是避免运动学shortcut。导航命令仍是条件。这与“显式数值状态越多越好”不同。

所以，本项目添加数值状态通道可以修补状态表达并提高误差表现，但若研究目标包括视觉/协同贡献，需要保留state-only、去图像和去路端的对照。不能把状态主导的低L2写成VLM推理能力证据。用户当前协议包含自车状态；本次研究不自动删除它。

## 4. 其他论文的训练证据与适用边界

### 4.1 UniAD / UniMM-V2X / UniV2X

- [UniAD本地PDF](references/openaccess/22_UniAD_CVPR2023.pdf)，第5页§2.4/2.5：ego motion query结合command，通过BEV交互规划；先训练tracking/map感知6epoch，再所有感知、预测、规划模块端到端20epoch。这是模块联合训练描述，不等于每一层基座参数必然全部解冻。
- [UniMM-V2X本地PDF](references/aaai/06_UniMM-V2X_AAAI2026.pdf)，第5页训练说明：40epoch感知预训练，20epoch端到端运动/规划；多任务目标，AdamW LR1e-4、weight decay0.01。它支持先建立表示再训练规划，不能用来证明冻结VLM的末token回归有效。
- [UniV2X本地PDF](references/aaai/15_UniV2X_AAAI2025.pdf)，第6页说明其比较设置去除基线ego velocity embedding，且提到附录分析。当前下载稿没有足够附录细节支撑完整冻结/阶段配方；不能把UniAD或UniMM的epoch安排直接当成UniV2X原文。FlowQueryNet部分也有独立训练边界，不宜泛称全部模块共同端到端训练。

### 4.2 DiffusionDrive

[本地PDF](references/openaccess/36_DiffusionDrive_CVPR2025.pdf)，第5页训练/anchor，第7页实现。

使用训练集轨迹构建20个k-means anchor，对有噪anchor进行分类与正样本轨迹重建；空间BEV和agent条件经cross-attention进入decoder。论文强调仅替换成常规扩散结构不足以利用空间条件。NAVSIM设置从零100epoch、batch512、AdamW LR6e-4，有成熟视觉规划结构与大数据预算。

可借鉴多模态anchor、空间条件和分类/回归组合。当前先建立有效单模态F0更合适；没有证据表明扩散一定优于本项目的小数据MLP。anchor必须只用train轨迹拟合。

### 4.3 SparseDriveV2

[原文](https://arxiv.org/abs/2603.29163)，[本地PDF](references/arxiv/26_2603.29163.pdf)，第8页规划接口，第13、18页不同基准训练说明。

它明确分开场景特征编码与ego状态编码，然后生成/评分路径与速度候选；支持数值状态拥有独立输入通道。NAVSIM与Bench2Drive的训练阶段、评分监督不同，不能拼成统一配方。它不是VLM末hidden→18维回归头的相同结构。

### 4.4 DriveVLM / Drive-R1

- [DriveVLM](https://arxiv.org/abs/2402.12289)，[本地PDF](references/arxiv/07_2402.12289.pdf)，第5页Dual、第8页训练：VLM驾驶推理/规划通过微调学习，Dual将慢速VLM轨迹交给快速传统或神经planner精化；额外数据共同训练用于保持泛化。不是冻结通用VLM读hidden的连续头方案。
- [Drive-R1 AAAI官方原文](https://ojs.aaai.org/index.php/AAAI/article/download/37602/41564)，第4页SFT/RL、第6页实现：InternVL2-4B先领域全参数SFT，再推理SFT，后续GRPO结合轨迹/动作/格式等reward；输出包含文本轨迹。与连续decoder不是相同训练路线，当前F0尚未有效，优先做RL缺乏必要基础。本地`40_Drive-R1_AAAI2026.pdf`无法正常解析，本次证据来自在线原文。

### 4.5 预测模型与空间VLM

- MTR/MTR++：意图query、动态query和多模态概率轨迹监督；MTR第7页有AdamW与30epoch训练细节。目标是其他交通参与者运动预测，不能直接当自车VLM规划训练证据。
- GameFormer：第4–5页level-k交互解码、GMM/NLL及交互损失；支持显式对象交互与多模态监督，非冻结VLM。
- MotionLM：第4页运动离散token和自回归训练；允许其生成任务定义内的teacher forcing。这里的问题不是全面禁止teacher forcing，而是本项目规划readout不能在训练中偷看推理时没有的未来答案。
- SpatialRGPT：第6、23页视觉/空间连接器对齐及instruction训练。可作为空间表达适配的依据，没有直接证明轨迹头训练策略。

## 5. 与当前F0的逐项对照

当前事实来源：[F0产物审查](f0_training_audit_20261004.md)、`tools/continuous/cache_features.py`、`model.py`及正式checkpoint。以下是工程核验，不是从论文推测。

| 环节 | 当前实现 | 文献带来的诊断方向 |
|---|---|---|
| VLM | 真正使用本地7B LLaVA，但完全冻结、no_grad、缓存hidden | 用到了VLM，不等于语言表示获得规划监督 |
| readout | final-normalized最后有效token；实际chat模板末尾是assistant起始后的换行 | 与专用受训waypoint query存在差距；不能称模型输出恒定 |
| 可训练部分 | F0有效训练约224.7万参数head，路端encoder不发挥作用 | 随机head缺少驾驶先验；泛化风险需实验验证 |
| ego状态 | 速度/加速度/历史作为文本进入同一hidden | 可能存在数值信息读取瓶颈，状态旁路是待验证修补 |
| 轨迹loss | 9×2位移累加，唯一masked cumulative-waypoint Smooth-L1 | 不是Omni flow matching，也不是AURORA完整规划目标 |
| 数据 | 929train、37train scenes；157val、7val scenes，scene不重叠 | 大模型或复杂decoder迁移收益不能由大数据论文直接保证 |
| 优化 | AdamW固定LR1e-3、weight decay0、100epoch，无scheduler | 可比较调度/正则，但没有证据认定单独LR就是根因 |

正式F0 best主平均L2=3.9697m，同val恒速1.4660m；best train/val=2.5155/3.9697m，last=1.9753/4.3042m。已见泛化差距。训练集拟合的数值状态线性探针val约0.6312m，冻结hidden探针约3.33–4.06m，支持通路问题的诊断假设。

探针不属于正式新基线；状态来源与时间因果性还需独立审计。高hidden余弦相似度也不能证明特征完全无信息。当前不能说“VLM没加载”“没有训练成功”“完全因为没LoRA”或“必然过拟合是唯一原因”。

## 6. 建议的实验顺序：先区分原因，再升级训练

这是研究建议，**没有修改用户已确认的M1/F0/F1/F2协议，没有启动训练**。升级后的实验应使用新版本/新目录；旧F0保留。

### 6.1 第一轮：低成本的通路和泛化实验

1. 正式建立state-only基线：直接编码数值速度/加速度/因果历史，归一化统计只从train拟合。审计状态/历史不是未来轨迹差分或未来GT推导。
2. 固定当前冻结VLM，比较最后token、明确定位的PLAN读出、少量任务query对观测token cross-attention。不同readout特征需要重新提取，不复用身份不匹配的旧cache；新token在完全冻结且无对齐时不保证更优。
3. 对同一结构比较有/无显式数值状态；可试恒速轨迹加残差，但把它标记为新的运动学先验，不冒充论文原版。
4. 采用小而预注册的优化对照，例如固定LR与warmup+cosine、少量weight decay；每轮只改清楚的因素，用validation选择，不重复查看test选型。

这个阶段回答“现有VLM表示能否被读出来”“是否数值信息通道限制精度”“改善是否仅来自状态”。即使state-only胜出，也不能据此删除视觉研究主线。

### 6.2 第二轮：AURORA启发的连续规划适配

若确认要改变首轮M1，可做一个明确的新planner版本：

- 使用观测末尾的规划任务token/query，确保因果readout；需要新增token时明确处理embedding可训练性与tokenizer保存。
- **阶段A：**冻结基座，先预热head/任务query/状态encoder，确认数值稳定、训练集可拟合。阶段时长由验证曲线决定，不机械复制论文。
- **阶段B：**冻结视觉主干和语言基座主体，开启语言LoRA与head联合训练，让轨迹loss回传到任务表示；LoRA和新head采用独立优化参数组，LoRA LR通常应比随机head更谨慎，具体值通过小规模对照确定。
- **阶段C（条件具备才做）：**混合真实驾驶QA/空间监督与规划样本，避免仅为轨迹拟合损害语言表示。不得编造QA任务或把未来GT答案送入readout。

这是由论文原则推导出的本项目方案，**不是AURORA原样复现，也不是已验证的最优训练配方**。929条轨迹不足以支持随意全量解冻7B；LoRA也可能过拟合，要比较冻结组、记录train/val差距和多个seed。

重要工程限制：现有缓存hidden已经detach，**不能在旧cache上训练出语言LoRA**。阶段B必须在线进行带梯度的VLM前向，或设计真正可微的分段表示；保存LoRA、head、任务embedding、tokenizer、状态归一化统计、基座身份和输入版本。不能仍跑CPU head脚本却把实验命名为“VLM联合训练”。

### 6.3 第三轮：有效F0之后再测协同

新版本先通过F0验收，再以相同架构、车端输入、初始化规则和训练预算分别训练F1/F2。不能拿升级LoRA的F1与旧冻结MLP F0比较并归因于路侧信息。

Omni方向可作为后续独立分支：冻结视觉patch encoder，训练多模态trajectory Transformer，先建立单车规划预训练再协同迁移。这会改变模型类别、预训练预算和语言研究定位，不能悄悄替代现有VLM主线。

### 6.4 必要消融与验收

| 对照 | 要回答的问题 |
|---|---|
| state-only vs image+GT text+state | 图像/对象信息是否贡献超出运动学预测？ |
| 冻结VLM+head vs LoRA+同构head | 语言侧规划适配是否有效？ |
| 当前末token vs任务token/query | 规划接口是否是瓶颈？ |
| 图像去除/错配；对象文本去除/错配 | 两条观测通路是否被实际使用？ |
| 完整state vs减弱state的诊断组 | 模型是否主要依赖ego状态？ |
| zero RSU vs独立GT RSU vs detector RSU | 在同构训练条件下量化协同价值与感知误差 |

固定权重删掉或错配输入属于敏感性/分布变化诊断；**要比较某模态的训练价值，还要分别训练对应消融组**。保留当前split和统一evaluator，报告L2@1/2/3s、三者平均、3s/4.5s FDE、coverage及有效数量。碰撞/道路约束只有真实checker和可用标注时才评测；目前未评测不能填写零。

## 7. 自车状态shortcut的补充原始文献

这篇不在当前本地25份PDF内，单独标为外部补充：[Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?](https://arxiv.org/abs/2312.03031)，CVPR2024，Zhiqi Li等。

作者指出nuScenes开放环模型可能过度依赖ego状态，低轨迹误差未必表明充分利用感知，并讨论道路约束评测。它支持增加状态基线和模态消融的必要性，不证明DAIR-V2X子集一定具有同等问题。结合Omni的主动去速度设计，对本项目的合理解释是：**状态旁路可以是工程改进，但视觉和协同贡献必须另外验证。**

## 8. 全部文件筛查与引用索引问题

| 分类 | 实际文件/工作 | 本次处理 |
|---|---|---|
| 直接规划/VLM训练 | 03 AURORA、06 UniMM、07 DriveVLM、15 UniV2X、17 Omni、22 UniAD、26 SparseDriveV2、36 DiffusionDrive | 深读相关方法/训练/附录 |
| 运动预测 | 19 cooperative trajectory/V2X-Graph、23 MTR、24 MotionLM、25 GameFormer、37 MTR++ | 核查decoder和监督；不当作完全相同的自车规划任务 |
| 空间VLM/视觉基础模型 | 32 SpatialRGPT、35 DINOv3 | 核查适配或冻结依据；无当前轨迹头直接实证 |
| 协同感知/数据集/综述 | 11 TUMTraf、12 V2X-Real、14 V2X-Radar、28 Where2comm、30 QUEST、38 competition progress | 全文筛查；主要帮助输入/通信/数据背景 |
| 协同控制 | 21 Coopernaut | 输出控制的模仿学习路线，不是VLM连续waypoint头 |
| 内部材料 | 02 opening report | 研究背景，不作为外部训练验证 |
| 无法正常解析 | 40 Drive-R1、39_2304.11821 | Drive-R1补读AAAI原文；39不用于本次结论 |

引用索引需要核对：

- AURORA真实标题为Roadside-Cooperative Autonomous Driving: From Data Platform to Vision-Language End-to-End Reasoning，作者Yitao Xu等；索引中的Xiang等作者信息不符。
- OmniV2X作者Juntong Peng等；实际文件名为`17_OmniV2X_DAIR-V2X-Seq.pdf`，它是规划器论文，不是仅介绍DAIR数据集的论文。
- `26_2603.29163.pdf`是SparseDriveV2，不是泛指UniAD类规划。
- MTR与MTR++为两篇不同工作，不能互换训练设置。
- Drive-R1与编号39文件虽然有PDF头，但解析报xref/trailer错误；本次未覆盖、删除或替换原文件。

提取文本暂存于`/tmp/v2x_paper_training_research_20261004/`，用于定位章节；长期证据应以原始PDF及以上原文链接为准。references.md是导航，不替代原文，也不能把历史摘要直接当作本次已核实事实。

## 9. 后续问答补充：E2特征对照与AURORA的plan机制

本节由AI辅助核查，使用academic-research-suite的source-verification角色在线执行，未进行独立多审阅者验证。对照本地原文§4.3（PDF第5页）与Appendix B（第15页），并核对[arXiv HTML](https://arxiv.org/html/2608.21032v1)。AURORA按当前核查记录是arXiv预印本；这里只核查方法描述，未独立复现实验或核实同行评议状态。

### 9.1 推荐增加一个先行对照，不直接跳到联合训练

之前E2 checkpoint-932目录有adapter_config.json与adapter_model.safetensors，基座为llava-next-interleave；F0 run.json的features.model指向基础llava-next-interleave，缓存入口没有加载E2 adapter。因此建议先比较：

- A：基础LLaVA冻结特征＋当前同构MLP（保留已有正式F0）。
- B：基础LLaVA＋E2驾驶LoRA，全部冻结后重新提取当前F0观测特征＋相同MLP。
- C：相同split的数值state-only基线。

A/B必须固定当前ego-only图像、独立车端GT文本、状态与prompt、readout、head初始化、训练预算和评测器，仅改变feature extractor权重。E2本来使用的输入来源与当前F0不同，应说明迁移带来的分布变化，并审计E2训练/选型数据与本次split关系；不能直接拿旧E2成绩当B组成绩。cache身份需要记录基座与adapter身份。

B改善支持驾驶适配表示有益；B仍差不能证明LoRA无效，因为训练生成接口与当前回归readout可能不匹配。随后做任务token/query和状态旁路对照，最后才训练在线语言LoRA＋head。逐步推进的价值是能够归因，不是宣称各步骤必然提高指标。用户当前询问方案是否合理，本次未启动上述实验。

### 9.2 原文明示的plan通路

车/路感知→CQAF对齐融合→检测/地图semantic tokens→替换VLM视觉占位→LoRA适配VLM→专用<wp>最终hidden→连续规划decoder。这里的“plan任务”是用未来轨迹及约束监督专门的输出接口，而非要求VLM先输出完整轨迹文本后再读取文本。

主要decoder为条件概率VAE：论文描述present/future Gaussian pair，采样latent code，再由GRU递推展开候选轨迹；其他decoder为确定性MLP直接回归future waypoints，以及去噪trajectory anchors的diffusion。VAE训练/推理时的prior/posterior具体输入、各层维度、采样次数和候选选择规则没有完整提供，不能写成已核实代码级配方。future GT只能用于训练标签/后验学习，不得进入部署时观测或本项目readout。

MLP/VAE规划目标为L_reg + L_bound + L_col，VAE加L_KL；L_reg按command条件监督轨迹，L_bound约束地图边界，L_col约束与融合agent轨迹的碰撞。语言目标为QA next-token交叉熵。阶段4和5总目标写为L_vlm + L_plan；不虚构原文未给出的具体回归形式、项权重或checker实现。

语言LoRA参与规划阶段训练，因此轨迹误差可通过decoder和h_wp反传到语言adapter；基座权重冻结不妨碍对adapter计算梯度，no_grad或离线detach缓存才切断这条通路。

阶段3先做视觉语言QA对齐，不监督planner；阶段4加入规划decoder与语言LoRA训练；阶段5混合QA/规划，QA-only只计算语言目标。正文阶段5“全部解冻”与附录Table9 backbone仍冻结的冲突继续保留，不能笼统解释为全参数语言微调。

<wp>的具体序列位置、新embedding训练细节与是否存在先生成推理再规划的固定模板未充分交代。把本项目PLAN放在当前观测之后、未来答案之前，是因果接口的实施建议，而不是已确认的AURORA原模板。head单独预热也是本项目建议；AURORA明示的是QA对齐后进入LoRA＋planner阶段。
