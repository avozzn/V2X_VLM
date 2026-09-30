# 路侧 Agent Token 改造与融合方案

> 日期：2026-09-30  
> 定位：DrivewithVLM 后续开发和实验设计文档；以下新增模块、配置名称均为建议，尚未实现。  
> 依据：[研究报告（7）](<deep-research-report (7).md>)、[实际进展与下一步建议](v2x_vlm_next_steps_20260930.md)，以及本地 OmniV2X / UniV2X 代码。  
> 当前决策：优先实现 **F3：VLM 后的对象 cross-attention + 连续 MLP planner**，先 Oracle，再替换冻结 D2 detector 输出。

## 1. 目标和参考代码

目标是让车端规划器接收路侧交通对象的物理状态，先验证结构化路侧信息是否有规划价值，再研究将它输入 VLM 内部是否带来额外收益。

OmniV2X 不是使用语言大模型的 VLM。其视觉编码器、车辆状态、路侧对象和可选地图作为条件输入，驱动 Rectified Flow / DiT 轨迹规划。借鉴的是对象表示、编码和条件融合机制，不是直接复用一个现成的 LLaVA 路侧接口。

| 可参考实现 | 本地文件 | 复用范围与适配边界 |
|---|---|---|
| 路侧 GT 对象 → 13D 状态、nearest Top-K、padding / mask | `/home/zzn/V2X_VLM/OmniV2X-main/navsim/agents/omniv2x/core/infra_bbox_features.py` 的 `V2XObjectFeatureBuilder` | 抽取数值编码逻辑；将 NavSim Scene 输入改成项目的 SPD 对象缓存；不能直接读取 cooperative 合并 GT 代替路侧观测 |
| 对象 MLP、source embedding、对象条件编码 | `/home/zzn/V2X_VLM/OmniV2X-main/navsim/agents/omniv2x/models/diffusion/dit.py` 的 `infra_bbox_encoder`、`_encode_infra_bbox_conditioning` | 可抽取轻量 MLP；其主 planner 是 DiT，需要重新接入 VLM latent / MLP planner |
| 规划 token 对条件 token 的 cross-attention | 同文件的 `UnifiedDiTBlock` | 借鉴 query/key/value 接口；新实现需显式传对象 padding mask |
| 路侧独立标注读取 | `/home/zzn/V2X_VLM/OmniV2X-main/navsim/common/dairv2x_converter.py` 的 `_load_infra_annotations` | 借鉴对象来源选择；坐标、速度和时间必须按当前项目协议审计 |
| 车路坐标对齐及 query 匹配 | `/home/zzn/V2X_VLM/UniV2X/projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py` 的 `AgentQueryFusion` | 参考已有标定矩阵约定；它不是可直接用于 LLaVA 的对象编码器 |

官方参考：[OmniV2X 源码](https://github.com/JuntongPeng/OmniV2X)、[对象特征构造](https://github.com/JuntongPeng/OmniV2X/blob/main/navsim/agents/omniv2x/core/infra_bbox_features.py)、[UniV2X 源码](https://github.com/AIR-THU/UniV2X)。

## 2. 所有融合方式共用的路侧接口

### 2.1 数据来源：Oracle 和 Detector 可互换

```text
路侧独立 GT boxes/tracks ── Oracle exporter ──┐
                                            ├─ 统一 Agent Cache ─ Agent Encoder ─ Fusion
冻结 D2 detector 输出 ── Detector exporter ──┘
```

Oracle 使用基础设施侧观测标注，不把所有合并 cooperative GT 当成路侧可见对象。RSU-only 指“路侧观测到而车端未观测到”，不是“坐标落入路侧 pc_range”。若仅有视锥/范围判断，记录为几何可见代理；没有遮挡证据就不标成真实 occluded。

Detector 使用推理输出，planner 不读取检测 GT。两种 exporter 产出相同合同、相同坐标和类别映射，planner 无需因来源切换而重写结构。

### 2.2 几何状态合同

第一版参考 OmniV2X 的 13D：

```text
[x, y, z, l, w, h, sin(yaw), cos(yaw), vx, vy,
 class_vehicle, class_pedestrian, class_cyclist]
```

统一数据接口另保留以下字段，按消融逐步启用；不要在第一轮同时测试所有机制：

| 字段 | 用途 | 缺失时的规则 |
|---|---|---|
| `score` / `score_valid` | Detector confidence | Oracle 可设 1，但不能解释为现实检测器可靠性 |
| `age_s` / `timestamp` | 消息年龄与实际时间对齐 | 使用真实配对时间；不能默认所有路侧消息 age=0 |
| `source_id` | 区分路侧来源 | 单 RSU 可固定；与 planner 的 token modality embedding 区分 |
| `velocity_valid` | 速度是否可用 | 未知速度数值可填 0，但有效性必须为 false |
| `state_valid` | 几何分量有效性 | 与整对象 padding mask 分开 |
| `track_id` | 时间关联、后续 memory | 作为关联元数据；不要默认将任意全局 ID 直接当可泛化 embedding |
| `uncertainty` / `uncertainty_valid` | 质量估计 | 第一版可不启用；不能把 score 直接当 covariance |
| `semantic_features` / `semantic_valid` | 后续检测 query 语义 | 第一轮 geometry-only 不启用 |

固定所有分支的尺寸顺序为 `[l,w,h]`。项目部分 box 接口使用 `[w,l,h]`，exporter 必须显式转换。Unknown 类别需明确处理；不要无条件套用参考代码中“未知类别归 vehicle”的映射。

### 2.3 Batch 合同

```text
agent_geometry:  [B, K, 13]     float32；第一轮 K=16
agent_mask:      [B, K]         bool；True 表示真实对象
agent_quality:   [B, K, Dq]     score/age/有效性等，按配置启用
agent_source:    [B, K]         integer
has_rsu_message: [B]            bool
history_xy:      [B, H, 2]
history_mask:    [B, H]
future_xy:       [B, T, 2]      未四舍五入 GT
future_mask:     [B, T]
future_times_s:  [B, T]         或同协议共享 [T]
token / scene / frame / timestamp / schema: 元数据
```

有效消息但零对象与通信丢失应在数据层区分。融合层可以对两者都退回 ego 分支；后续若加入 message-status embedding，要独立消融。

### 2.4 坐标和时序

- 全部对象和 ego trajectory 使用当前车端 LiDAR 坐标，X 前、Y 左；heading 表示、矩阵乘法方向统一。
- 中心位置做刚体变换，速度做旋转，yaw 通过方向向量变换后恢复。协方差按同一参考系旋转传播。
- 有延迟时，先在一致的世界或明确的参考系中做运动补偿，再变到当前 ego frame，避免将不同时间的局部坐标直接相加。
- 明确速度是目标的绝对速度还是相对 ego 速度；不要把 ego 运动当成目标运动。速度只能来自截至消息时刻可得的状态或轨迹。
- 缓存带 frame 和 timestamp，不允许靠图片路径碰巧一致隐式匹配。选择和补偿不使用未来 GT。
- 以训练集统计或固定物理尺度归一化，保存参数并复用到 val/test；不对未知分量当作真实零值训练。

## 3. 共八种可尝试的融合结构

以下是本项目的八个实验选项，不是对所有融合方法的穷举。F1～F7 可围绕同一 geometry 输入比较；F8 属于依赖重检测架构的参考路线。K=16 是首轮预算选择，不是最优性的结论。

记 `h` 为车端观测上下文产生的 `[PLAN]` hidden state，`A=Encoder(G)` 为对象 token。`Pool` 必须考虑 mask，并对空集合返回确定的零向量。

| ID | 融合方式 | 路侧信息进入位置 | 是否进入 VLM 内部 | 改动量 | 建议定位 |
|---|---|---|---|---|---|
| F1 | Masked pooling + 拼接 MLP | VLM 后的 planner 输入 | 否 | 小 | 最小融合对照 |
| F2 | Pooling + 门控残差 | VLM 后的 planning latent | 否 | 小 | 质量门控对照 |
| **F3** | **PLAN → 对象 cross-attention** | **VLM 后的 planning latent** | **否** | **中** | **首个主实现** |
| F4 | 多个轨迹 query 的对象交互 decoder | 连续 trajectory decoder | 否 | 中 | 比较单 latent 与逐时间步交互 |
| F5 | 对象 soft-token 插入 VLM 输入 | 语言模型输入 embedding 序列 | 是 | 中高 | F3 有效后的主升级 |
| F6 | VLM 层内 cross-attention adapter | 若干语言模型层 | 是 | 高 | 有额外收益证据后尝试 |
| F7 | VLM 对象输入 + 几何旁路 | VLM 输入及连续 planner | 是 | 中高 | 检验语义与几何双路径 |
| F8 | UniV2X 式 BEV/query 融合 | 检测/BEV 层，随后接 planner | 可选 | 高 | 现有 D3/外部感知融合参考 |

### F1：Masked pooling + 拼接 MLP

```text
ego image/history -> VLM -> h ─────────┐
                                     ├-> concat -> MLP -> displacement
RSU objects -> Encoder -> masked pool ┘
```

`z = MLP(concat(h, Pool(A)))`。保留每对象编码，再做 masked mean；可将 masked max 作为后续小消融。第一版保持简单。

优点：最容易实现，帮助确认“增加对象数值是否就有收益”。限制：池化丢失对象级区别，不能对不同风险对象进行显式选择。

空对象时 pool 返回零，避免除零。零向量输入不保证网络等同于独立 ego-only checkpoint；仍需评测。

### F2：Pooling + 门控残差

`a = Project(Pool(A))`，`g = sigmoid(Gate(h,a,quality))`，`z = h + has_agent * g * a`。

Gate 第一版可以是标量或通道向量，输入 pooled quality 不使用未来标签。门控用于学习何时接收路侧上下文，不能直接把 gate 当成已校准信任概率。

优点：保留 ego 表示，有明确开关；适合 age/confidence 消融。限制：仍然有池化的信息损失。

空对象时强制增量为零。若 adapter 输出严格零初始化，其上游 encoder 首步可能无梯度；需检查梯度路径，或采用小幅初始化加 gate。

### F3：PLAN → 对象 cross-attention（推荐第一版）

```text
ego image + causal history/state -> VLM -> PLAN latent (query)
                                                   │
RSU states -> Agent MLP -> agent tokens (key/value) ─┤
                                                   ↓
                                  masked cross-attention + residual
                                                   ↓
                                         MLP -> Δxy -> cumsum
```

`c = CrossAttention(Q=h[:,None,:], K=A, V=A)`，`z = h + gate * c`。对象 encoder 可先使用两层 MLP；fusion 维度可先投影到较小维度，再投影回 VLM latent 维度。

推荐可配置起点：K=16、fusion_dim=256、4 heads、1 fusion block、T=9。这些只是初始工程选择，需要验证，与 OmniV2X checkpoint 维度没有直接兼容保证。

优点：对象不会在 attention 前被池化；改动局限于 VLM 后，可以明确隔离融合层的贡献。限制：路側信息尚未参与 VLM 内部推理，因此只能称为“VLM 特征驱动的对象条件规划”。

实现关键：

1. valid mask 的 True 表示存在对象；PyTorch `key_padding_mask` 的 True 表示需要屏蔽，所以传入 `~agent_mask`。
2. 用 `nonempty=agent_mask.any(dim=1)` 划出非空样本，只对这些样本调用对象 attention；其余样本的增量为零。
3. 对空样本直接保留 ego latent，不要先计算全 mask attention，再乘 0；NaN 乘 0 仍是 NaN。
4. 不依靠对象向量置零屏蔽 padding：零向量仍可能进入 softmax 分母，投影 bias 也可能产生非零 key/value。

这一结构是首轮 Oracle / Detector 对照的推荐实现。

### F4：多个轨迹 query 的对象交互 decoder

构造 T 个 learnable planning queries，加时间 embedding，使用 ego latent 调制或作为额外 context。Decoder 对对象 tokens 做 cross-attention，并可在轨迹 queries 之间做 self-attention，最后每个 query 输出一个 displacement。

```text
T planning queries + time embedding
           ↓
ego conditioning + agent cross-attention + query self-attention
           ↓
       T × 2 displacement
```

优点：不同未来时刻可关注不同对象；相比单 PLAN latent 有更丰富交互。限制：引入 decoder 容量，改善不能全部归因于融合形式。

需额外做同 decoder 的 ego-only 对照。时间 queries 是预测变量，可以互相交互；但不允许把未来 GT 或目标动作作为 decoder 输入。空对象时跳过对象 attention，保留 ego-conditioned query 路径。

### F5：对象 soft-token 插入 VLM 输入

将每个对象向量投影到语言模型 embedding 维度，作为连续 soft-token 放入语言模型上下文，而不是转成坐标文字。

```text
[ego visual tokens] [history/state context] [RSU object tokens] [PLAN]
                                  ↓
                                 VLM
                                  ↓
                              PLAN hidden
                                  ↓
                            continuous MLP
```

在因果语言模型中，`[PLAN]` 必须位于所有需要读取的观测 tokens 之后。若保留语言辅助答案，答案放在 PLAN 后方，规划 readout 不能读取 teacher-forced 答案信息。

优点：VLM 内部可以结合视觉和对象状态形成 planning latent。限制：涉及 LLaVA 图像 token 展开、`inputs_embeds`、`attention_mask`、`position_ids`、labels 对齐；不能仅在原始 `input_ids` 上拼一段 tensor。

padding 对象不应算作有效上下文。无对象时，优先构造与 ego-only 相同的观测序列；若使用固定 padding 长度，要正确处理 attention 和位置编码，不要默认两种长度行为相同。

先验证底层 `language_model`、图像 embedding 合并和 wrapper 的实际接口，再设计注入。新增 `[PLAN]` token 时同时保存 tokenizer 和 embedding 参数；LoRA-only 保存不一定覆盖新增 token embedding。

### F6：VLM 层内 cross-attention adapter

在选定的语言模型层插入残差 cross-attention，让语言/视觉 hidden states 查询外部 agent memory。先只选后部少数层，冻结基础模型，训练 adapter 和可选 LoRA。

优点：对象 context 可反复参与表示更新，不必把每个对象插入输入序列。限制：依赖具体模型结构；需处理 gradient checkpointing、混合精度、checkpoint key 和可能的 KV cache 一致性。

若仅一次 forward 读取 PLAN，不必立即支持自回归 KV cache；若仍输出语言答案，需明确缓存策略。所有 adapter 都必须支持对象 padding 和空集合；空集合时残差增量为零。

这是适配器架构实验，第一轮不做，以免工程复杂度掩盖对象表示的价值。

### F7：VLM 对象输入 + 显式几何旁路

结合 F5 与 F3：对象 tokens 输入 VLM，同时同一批 geometry tokens 绕过 VLM，由连续 planner 直接读取。

```text
objects -> token projector -> VLM -> semantic PLAN latent ─┐
                                                        ├-> fusion -> trajectory
objects -> geometry encoder -> planner agent memory ────┘
```

目标是让 VLM 表示空间交互和意图，同时给连续 head 保留精确数值通道。第一版两个分支都来自相同 geometry；后续再加入 detector semantic query，避免把“新增语义输入”和“新增融合路径”混成一个实验。

必需对照：仅 F3、仅 F5、F7 双路径，并尽量控制参数量。风险：几何旁路可能让 VLM 对象分支被忽略，因此应分别移除每条路径，检查预测和关键子集性能。

空对象时两条路侧分支都关闭。只有 geometry bypass 改善，不能据此宣称 VLM 学会对象交互推理。

### F8：BEV/query 层融合后接规划

使用 UniV2X 的车路检测 query 对齐、匹配、融合或 BEV 协作，再将融合输出送入规划模块。已有 D3 可按这条路线继续作为诊断。

优点：复用现有 query 协同体系，可保留感知阶段融合。限制：强依赖 detector、pc_range、query semantics、时序状态和训练预算；高维 query 也改变通信量。

F8 不应阻塞 F3/F5。与轻量对象消息比较时，要分别报告 detector 能力和通信 payload，不把它当作只换一个 attention 的同预算消融。也不要无适配混搭不同模型的 query。

## 4. 两个独立对照：不计入八种结构融合

| 对照 | 实现 | 用途 |
|---|---|---|
| Direct-VLM continuous | 车端图像 + 路侧原始图像 → 同 VLM → 同连续 head | 判断对象表示是否优于直接视觉输入；单独报告传图通信量 |
| Object-text continuous | 同一批路侧对象 → 固定格式文字 → 同 VLM → 同连续 head | 判断连续 soft-token 是否优于数值文字表达；控制对象来源和 K |

现有 E2 的 perception 文本来自 GT，不能直接充当纯图像 Direct-VLM 对照。所有对照去除未声明的协同 GT 文本；ego-only 的对象输入若保留，也须来自可用的车端感知。

## 5. 连续 planner 和训练约束

- 第一版保持 T=9、dt=0.5 s，以便与现有 4.5 s 模型对比；主报告从同一预测提取 1/2/3 s，4.5 s 指标单列。
- displacement 为相邻未来时刻的位移，第一段相对当前原点；通过 cumsum 得到累计 waypoint。
- 先用 masked waypoint Smooth-L1 + 最后有效点 loss。若加 displacement loss，其 target 需有连续有效区间；不能在缺失中间点上直接差分。
- velocity/acceleration regularization 使用实际时间间隔和有效性。displacement 表示自身不能保证轨迹运动学可行。
- collision loss 在可靠 occupancy、坐标和时间对齐后再加入；碰撞 GT 缺失不得记为零碰撞。
- `[PLAN]` 位于观测后、答案前；不使用 GT risk/action、future motion、future route 作为推理输入。它们可以作为训练监督。
- Ego-only checkpoint 是共享起点。初期冻结视觉/语言主体，训练 fusion/head；VLM 内部注入方式若需适配，可逐步启用 LoRA并报告训练预算差异。
- RSU dropout 应真正丢消息/对象，第一轮将其作为独立设置。处理空输入正确不等于自动具备不退化的 ego-only 性能。

## 6. 项目代码改动清单

建议按模块新增，而不是把路侧逻辑散落在 prompt transforms 中。以下名称尚未创建：

| 模块/文件建议 | 内容 |
|---|---|
| `tools/data/export_rsu_agent_cache.py` | Oracle 路侧状态导出、按 frame 配对、坐标/时间转换、缓存合同 |
| `tools/data/export_detector_agent_cache.py` | 冻结 D2 推理导出；boxes/score/velocity/query 有效性；不读取 GT 输入 |
| `projects/Robodrivevlm/model/agent_encoder.py` | geometry MLP、quality/source 编码、统一 norm |
| `projects/Robodrivevlm/model/agent_fusion.py` | F1/F2/F3/F4 模块及空输入处理；同一接口切换 |
| `projects/Robodrivevlm/model/vlm_agent_adapter.py` | F5 的 soft-token 合并和 F6 的层内 adapter；后续阶段实现 |
| `projects/Robodrivevlm/model/continuous_planner.py` | VLM wrapper、PLAN readout、融合调用、displacement head、loss |
| `projects/configs/Robodrivevlm/MMdrive_v2x_continuous_*.py` | 独立实验 config：ego/direct/oracle/detector、fusion_id、K、schema |
| 现有 `spd_vehicle_e2e_vlm_dataset.py` | 按 token 读取结构化 targets 和 agent cache，传递 masks；不只保留 QA 字符串 |
| 现有 `tools/collators/llava_interleave.py` | 正确处理整个 batch；目前 `instances[0]` 不能支撑 batch>1；补 agents/trajectory 批处理 |
| 现有 `tools/loaders/llava_interleave.py` | 保留旧文本 loader，新增 planner wrapper 加载方式 |
| 现有 `tools/robollm/train.py` / `train_utils.py` | 注册模型、规划 loss、数值 validation、按 validation 规划指标选 best、保存恢复全部新增权重 |
| 数值 inference / evaluator adapter | 输出同一 token/coordinate/times/trajectory 合同，复用统一评测，不让不同模型用不同 GT |

保存产物应包括：LoRA/需要训练的 backbone 参数、AgentEncoder、Fusion、trajectory head、新增 token embeddings、tokenizer、归一化尺度、数据 schema、训练 config 和 split manifest 引用。

不要求将 OmniV2X 的 NavSim/Lightning 全部迁入现有 HF Trainer；优先抽取独立 PyTorch 小模块，并保留来源说明。模型结构改变后，原 OmniV2X checkpoint 不会自动与 VLM wrapper 兼容。

## 7. 实验顺序：不要跑八种 × 所有字段的全排列

### 第一轮：判断 geometry 路线是否成立

固定 split、backbone、history/state、head、K=16、简单 nearest 选择和同样的训练预算，先跑：

| 实验 | 输入 | Fusion | 目的 |
|---|---|---|---|
| R0 | Ego only | 无 | 连续规划下界 |
| R1 | Ego + RSU image | Direct-VLM | 直接视觉对照 |
| R2 | Ego + Oracle geometry | F1 | 最小结构化融合对照 |
| **R3** | **Ego + Oracle geometry** | **F3** | **首个对象 attention 模型** |
| R4 | Ego + Detector geometry | F3 | 量化真实感知误差传入 |

需要量化总体和 RSU-only/交互子集。无明显增益时检查：对象覆盖和来源、坐标/时间、mask、梯度、输入冗余、训练不足和子集样本量；不能只凭总体 L2 无提升就断言协同无价值。

### 第二轮：比较融合位置

以 Oracle geometry 固定对象质量，优先 F3 vs F5。可视资源加入 F2 和 F4；F7 在 F5 训练/推理链路稳定后做。F6/F8 留作后续扩展。

F4/F7/F6 参数量可能更多，应报告 trainable parameters，并为 decoder 变化建立对应 ego-only 对照。几何维度和质量字段在这轮保持一致。

### 第三轮：表示和鲁棒性

在已选择的结构上，顺序加入 age、因果速度补偿、uncertainty、semantic feature、risk Top-K；每次改变一个机制。K=4/8/16/32、delay=0/100/300/500 ms、dropout=0/10/30/50%，先以少量条件筛选，再补完整关键曲线。

Delay 实验读取真实过去消息，不允许仅改 age 数值而仍使用当前对象 GT。Risk selector 使用当前状态或可部署 route / ego-only candidate；不使用 future GT。记录 geometry/semantic 分别序列化的 bytes 和消息频率，不能只报 token 数作真实带宽。

### 指标与结果模板

| Variant | Object source | Fusion | K | L2@1/2/3 s | Avg L2@1/2/3 s | FDE@3 s | Collision | RSU-only L2 | Coverage | Bytes/s | Latency | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ego only | — | — | 0 | | | | | | | 0 | | |
| Direct-VLM | RSU image | visual input | — | | | | | | | | | |
| Oracle geometry | GT RSU | F1 | 16 | | | | | | | | | |
| Oracle geometry | GT RSU | F3 | 16 | | | | | | | | | |
| Detector geometry | D2 prediction | F3 | 16 | | | | | | | | | |
| Oracle geometry | GT RSU | F5 | 16 | | | | | | | | | |

RSU-only 指标应基于固定的场景/样本 subset manifest，并注明定义与样本数。主模型和关键对照做多 seed 或 scene-level bootstrap，避免把同 scene 相邻帧当成独立样本夸大显著性。4.5 s FDE、off-road 等扩展指标在其 GT 支持后另报。

## 8. 验证与准入条件

### 工程验证

1. 混合 batch 包含有对象、零对象和消息丢失样本：forward/backward 有限，空样本路径正确。
2. 修改被 mask 的对象数值不影响输出；新增 padding 不应改变有效对象 attention。
3. 对无 positional agent 编码且无截断变化的 F1/F2/F3，打乱对象顺序应近似保持输出；F5/F6 不预设严格置换不变。
4. batch=2 两条样本都参与 loss，trajectory/agent mask 对齐；不再静默取第一条。
5. 保存加载后预测一致；同时恢复新增 head、encoder、tokenizer 和 norm。
6. Oracle 投影可视化和 RSU→ego 数值闭环通过；速度/尺寸/yaw/time 有可追溯合同。
7. PLAN readout 在训练和推理使用相同观测；任何 teacher-forced 辅助答案不能影响规划 latent。
8. 小样本可过拟合，确认 agent encoder、fusion、trajectory head 得到预期梯度，再启动完整实验。

### 研究验证

- 删除关键对象、打乱 RSU scene、移除所有 RSU，比较轨迹变化及误差/安全变化；仅有 attention 热图不构成利用路侧信息的充分证据。
- Oracle vs Detector 的差距应在同 planner 下量化；不能一边换数据来源一边换 encoder/head。
- VLM 内融合 F5/F6/F7 需要与同 geometry 的 F3 对照，才有依据讨论 VLM 内部交互是否值得。
- 所有结果限定在自己的统一协议和输入条件下；不会因为用了同 evaluator 就自动消除外部基线的预训练、split、GT/route 输入差异。

## 9. 推荐决策

**先做 F3，同时用 F1 验证简单数值融合的收益；Oracle 有可解释收益后替换 D2。然后比较 F5 与 F3，决定是否将路侧对象输入 VLM 内部。**

F2/F4 是成本较低的结构消融；F7 是语义与几何双路径的后续候选。F6 和 F8 不作为首轮必做项。semantic query、risk、memory、Flow 各自是后续扩展，不与第一版融合同时堆叠。

该顺序分别回答三个问题：路侧对象是否有用、真实检测误差损失多少、VLM 内部参与交互是否带来额外收益。通过这些证据再决定最终论文架构。
