# 基于研究报告（7）和实际代码进展的下一步建议

核查日期：2026-09-30。本文初始内容是开发建议；2026-09-30 已开始落地第一阶段数值合同与评测器，尚未重生成数据、训练模型或启动 GPU 任务。依据为 `deep-research-report (7).md`、现有两份优化计划、E2 validation 产物、E5 诊断报告和当前源码/日志。

## 当前状态与证据

| 模块 | 已核实进展 | 尚未完成或不能据此推断 |
|---|---|---|
| 数据 V2 physics | train/val/test=929/157/480，scene 无交集；生成器以真实 timestamp 重建历史和未来 | 这是内部子集，不能直接与其他论文官方样本集比较；目前过滤了未来不足 4.5 s 的帧 |
| E2 文本规划 | validation 选择 checkpoint-932，157/157 可解析；9 步平均 L2=1.1442 m，4.5 s FDE=3.2208 m | test 目录未发现评测产物；未证明路端图像、VLM 分别贡献多少 |
| 检测 D0 | 已记录车路坐标闭环审计 | 闭环正确不等于投影、对象来源和时间同步全部正确 |
| D1/D2 数值链路 | NCCL NVL 绕过方案、velo_update FP32 和 D2 动态 scaler 已完成短程验证 | smoke 不能替代 detection AP 或长训练稳定性 |
| D1 全量训练 | `work_dirs/e5a_d1_d2_q3_full_8gpu_e30/vehicle/` 已有 epoch_1～5；检查时 JSON 日志最新为 epoch 6、iter 170，loss=27.71683、grad_norm=136.8689 | 日志快照不保证进程当前仍运行；未发现这次 D2 全量 checkpoint 或 detection validation |
| 新模型 | 当前 loader 仍加载 `LlavaForConditionalGeneration`；新增 standalone `ContinuousTrajectoryHead` 与 mask-aware losses | head 尚未接入 VLM `[PLAN]` latent、batch collator、训练/保存加载流程；Agent Token 接口未实现 |

E2 每步误差换算为主时域：L2@1/2/3 s=0.0924/0.4797/1.2398 m，三者平均为 0.6040 m。这个数值仅适用于内部 157 帧 validation 和当前输入条件，不能与 OmniV2X 等公开数字直接比较。

只用同一 validation JSON 中的四舍五入 vx/vy 做恒速度预测 `p(t)=v*t`，157 帧全部纳入：9 步平均 L2=2.6191 m，L2@1/2/3 s=0.2189/1.2087/2.9706 m，4.5 s FDE=6.9628 m。该只读计算说明 E2 优于这一简单运动学对照，但不能证明其改善来自路端信息。

## 优先改进的具体代码边界

### 1. 协议和输入合同先固定

- 保留现有 V2 physics 作为旧实验快照。新建结构化数据版本，不覆盖已有 checkpoint 的标签和评测依据。
- `tools/generate_v2x_planning_v2.py`：已改为同时保存未取整的 `future_xy`、`future_mask`、`future_times_s`、history、ego state 及速度有效性；旧数据尚未重生成，且仍保留 4.5 秒完整未来筛选。
- `tools/eval/continuous_trajectory_metrics.py`：已新增模型无关数值 evaluator；当前 `trajectory_diagnostics_v2.py` 继续负责旧版文字动作诊断。数值 evaluator 尚需用真实 C0 prediction 文件做端到端验证。
- `projects/Robodrivevlm/model/continuous_planner.py`：已新增 `[B,H]` latent → `[B,T,2]` displacement → cumulative waypoint head，以及 masked waypoint、final-valid-point 和可选 displacement losses。CPU 单测验证输出形状、batch=2 两个样本都能反传、无效点被 mask；还未接入 LLaVA 或训练器。
- 统一输出应包含 token、frame、timestamps、numeric trajectory；报告逐时域有效样本数、缺失/非法预测数、L2@1/2/3 s、三时域平均、3 s FDE。4.5 s FDE 单列，不能混用 FDE 含义。
- 碰撞接入同一坐标和时间对齐的 occupancy checker；当前 diagnostics 的 collision 参数是可选输入，缺失时为 null，不能解释成零碰撞。
- 3 s 主协议可保留更多可用数据；用独立版本和 manifest 实施，再重新训练/评测必要对照，不能只改新模型的数据筛选。

输入需显式区分三个 setting：原始图像、oracle 对象状态、预测对象状态。当前生成器的 `perception_text(info)` 从 GT instances 产生位置和速度描述，再经 `QA_pairs[0]` / `Load_EGO` 进入输入。因此现有 E2 不能当成纯双图像 baseline。可视化 transform 的 GT 框绘制是独立读取、保存，不能仅凭它存在就判定图像输入被修改。

Ego-only 必须去掉 RSU 图像和协同 GT 对象文本；若保留 ego 感知对象，必须证明它们来自可部署的车端感知。Direct-VLM 应与 Proposed 使用相同 backbone、history、可用 route、连续 head 和训练预算。

### 2. 连续规划头优先于 lane head 和 D3 融合

建议先建立：`ego image + causal history/state -> VLM PLAN latent -> MLP -> [B,9,2] displacement -> cumsum -> waypoint`。

建议新增 `projects/Robodrivevlm/model/continuous_planner.py` 和独立实验 config，复用现有视觉模型加载方式。暂时不接 detector、flow 或大规模多任务网络。

需要同步改动：

| 路径 | 改进内容 |
|---|---|
| `projects/Robodrivevlm/datasets/spd_vehicle_e2e_vlm_dataset.py` | 按 token 对接结构化样本；保留轨迹、history、mask 和对象来源，不仅传 QA 文本 |
| `tools/collators/llava_interleave.py` | 当前只用 `instances[0]`，batch>1 会丢样本；新版要正确批处理可变文本/图片/agents，并传递数值 targets |
| `tools/loaders/llava_interleave.py` | 增加 planner wrapper 的独立加载入口 |
| `tools/robollm/train.py` / `train_utils.py` | 接入 trajectory loss、数值 validation、规划指标选 checkpoint；目前默认 best 指标是 eval_loss |
| 保存和加载路径 | 同时保存/恢复 LoRA、trajectory head、agent encoder、tokenizer 与归一化合同；不能只保存 adapter |

`[PLAN]` 放在输入上下文末尾，提取其 hidden state；训练和推理都能看到相同的观测上下文。禁止从含 GT trajectory/action 回答的末尾 hidden state 回归轨迹；即使保留 teacher-forced 语言辅助 loss，也要让 planning readout 不依赖未来答案。

第一版损失只用 masked waypoint Smooth-L1 和最后有效点 loss。可附加小权重 displacement loss。平滑/碰撞/action loss 逐项加入；displacement 不能自行保证运动学可行性。必须用实际有效时间间隔计算速度/加速度。

验证应覆盖：batch=2 两条样本均参与 loss、mask 不影响无效点、训练/推理 planning 输入一致、checkpoint 保存加载预测一致，以及小样本能过拟合。随后在冻结 split 上比较 ego-only continuous、同输入 E2 continuous、Direct-VLM continuous。

### 3. Oracle Agent Token 验证与检测训练可以独立推进

先用真正路端可观测的 boxes/tracks 构造 GT RSU tokens；不能把所有 cooperative GT 当作路端可见目标。若只有范围/视锥测试，应标注为几何可见代理，不能声称真实遮挡标注。

第一版字段：`xyz, length/width/height, sin_yaw/cos_yaw, vx/vy, class, source, age, state_valid, agent_mask`，先固定 K=16，用 nearest 或 confidence 作为简单 selector。缺失速度/协方差必须带有效性标记；GT token 的 score=1 不表示现实检测器置信度可用。

建议独立建立 token exporter、`agent_encoder.py`、`agent_fusion.py`，用 MLP/小 Transformer 编码状态、cross-attention 接入 planning latent。与 ego-only 比较关键场景指标和移除路端对象后的预测变化。不要预设 oracle 一定改善所有 overall L2：无改善时先排查信息冗余、样本覆盖、对齐和融合是否生效。

检测训练已有成果继续保留，D1/D2 接下来优先补 validation 和导出：boxes、score、class、velocity 及有效性、timestamp、track ID、可选 query feature。统一尺寸顺序（源码中同时存在 w/l 和 l/w 约定）、yaw 和速度旋转。冻结检测器生成缓存，先训练 adapter/head，后续才考虑联合微调。

当前全量脚本使用 `--no-validate`，训练 loss 下降不能选出最好的 token 来源。路端验证必须使用路端 GT 和正确坐标。以可用闲置 GPU 或训练结束后的调度做验证；本次审查没有启动额外 GPU 作业。

D3 直接 query/BEV fusion 留作诊断 baseline，不作为 Agent Token planner 的前置条件。也不建议为了完成旧 E3 编号，强制先开发全套 lane segmentation head；只有 route 来源可部署或转弯诊断明确指向道路信息不足时再加。

### 4. 逐步增加论文核心机制

顺序：geometry-only -> detector geometry -> geometry+semantic -> age/causal compensation -> uncertainty -> risk Top-K -> 可选 agent motion/flow。

- 时间补偿在同一参考系中进行，使用过去消息和截至当前可获得的速度；不能用当前 GT 重建延迟消息。velocity 是绝对目标速度还是相对 ego 速度必须固定。
- covariance 经坐标旋转传播；detector score 不等于已校准 covariance。先明确不确定性来源，再比较 confidence-only 与 calibrated uncertainty。
- risk selector 只使用当前/历史状态、可部署 route 或 ego-only candidate，不允许使用 ego future GT 做 Top-K。nearest/confidence/random 为必要对照。
- RSU-only/visibility 若只能由标注得出，优先用于监督和评测分组，推理输入改用可计算的检测匹配/可见代理。
- motion GT 可作为训练 target，推理只能使用预测 motion；初期使用恒速度 motion hypothesis 即可，不必立即实现 MTR。
- 删除关键对象、打乱 RSU scene、移除所有 RSU，是输入依赖诊断；轨迹变化还需结合安全和 GT 误差解释，不能仅凭敏感度宣称模型正确推理。
- 带宽计算以实际序列化 bytes × message frequency 为准，分别报告 geometry 和高维 semantic query；保留 latency、K、dropout/noise 曲线。

## 推荐开发顺序与验收

| 顺序 | 可交付改动 | 进入下一步的条件 |
|---|---|---|
| 1 | 输入来源审计、数值 trajectory/mask 合同、统一 evaluator；补 E2 test 最终报告 | 合同与 evaluator 代码已起步；重生成结构化 split、跑指标回归、完成 E2 test 后再准入 C0 |
| 2 | Ego-only continuous 与同 backbone Direct-VLM continuous | head 单测已通过；仍需 ego-only 输入数据、`[PLAN]` latent wrapper、batch collator、数值训练/保存加载闭环，再与旧模型按同输入条件比较 |
| 3 | Oracle geometry token + 简单融合，同时补 D1/D2 detection validation/export | 路端 token 来源/坐标/时间可追溯；能解释协同收益或无收益 |
| 4 | 冻结 D2 detector 替换 oracle，保留相同 planner | 定量分离检测误差与规划器能力 |
| 5 | geometry+semantic、age、uncertainty、risk selector 分项消融 | 相同输入预算下有验证集收益，且关键子集有足够样本 |
| 6 | flow、PDMS、外部数据或真闭环 | 主模型和必要消融已完成，不阻塞论文主线 |

主基线优先：ego-only、Direct-VLM、OmniV2X、Proposed；UniMM-V2X 按 checkpoint 可用性/训练资源加入，UniV2X 利用现有实现做参考。报告（7）要求三套外部基线，比 9 月 28 日计划更重，不必全部作为第一轮开发准入条件。

OmniV2X 官方公开了 DAIR checkpoint 和推理/PDMS 路径，适合先做独立复现。但其本地 converter 存在从未来轨迹推导 command、未来不足时补最后 pose 等行为；这需要在统一协议 adapter 中明确处理。重新使用同一 evaluator 并不能消除训练数据、预训练、输入标注、GT 构造和 route 条件差异。至少报告官方复现 setting 与内部统一 setting，检查训练场景是否与内部评测集重叠。

相关官方来源：[OmniV2X](https://github.com/JuntongPeng/OmniV2X)、[UniMM-V2X](https://github.com/Souig/UniMM-V2X)。研究报告中无法直接解析的旧 `turn...` 引用不能单独当作新颖性依据；论文创新需要回到原文核查。当前可验证的研究目标应是：在一致输入和通信预算下，带时间与质量信息的 Agent representation 是否改善协同规划，以及 VLM 是否为关键交互带来额外收益。
