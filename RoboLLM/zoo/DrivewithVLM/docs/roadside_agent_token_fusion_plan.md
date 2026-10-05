# 车路对象 Query、自车状态编码与 VLM 融合实施计划

> 更新：2026-10-05。第 1、3 节是当前设计和实验排期；第 2 节按用户要求保留原文；第 4 节保留系统/代码说明和真实实验记录。
> 已实现：T1对象文本和I1/S1双端共享ObjectEncoder＋语言LoRA自回归接口，真实7B训练及480帧test已完成；见3.1.1。未实现：StateEncoder、动态范围、I2/I3视觉/空间queries与VAE＋GRU。
> 最新用户决定：首阶段保持 VLM 自回归生成轨迹坐标文本，先研究双端感知与 VLM 输入。连续 MLP/VAE head 推迟；现有 online_model.py 是连续 head 的历史工程基础，不是当前直接生成入口。所有路侧信息必须进入 VLM。

## 1. 目标和参考代码

### 1.1 总体目标

面向带宽受限、视角不同且存在通信延迟的车路协同规划，构建两条互补的 VLM 输入：**周围交通参与者的空间表示**和**自车状态/因果历史表示**。保留车端原始图像，先证明独立路侧信息具有规划价值，再研究稀疏视觉 queries、时空适配与历史鲁棒性。

开题中的“稀疏感知—视觉重构”拆为可验证的步骤：先获取独立对象观测，再做自车相关对象选取和车路时空对齐，最后形成 VLM 可读的对象/空间 queries。第一阶段直接输入 soft tokens；在车端底图叠加几何提示的视觉重构作为后续对照，与 query 接口比较。框投影或提示叠加只能提供几何信息，不能称为恢复了路侧背景纹理或长尾语义。

**LiDAR 与 OmniDrive 的边界：**本地 OmniDrive 的 `mask_eva_lane_det_vlm.py` 配置 `use_lidar=False`。`petr3d.py:position_embeding` 利用深度采样和 `lidar2img` 的逆变换构造 3D 位置编码；LiDAR 坐标系、3D 标注监督不等于推理时输入点云，也不等于已经实现“LiDAR 实测深度引导的图像→BEV”。本项目可借鉴其 3D/时序位置与视觉 carrier queries。开题提出的 LiDAR 深度引导另列为有实际传感器/标定/深度输入支持后的多模态扩展，不能作为 OmniDrive 已有实现写入首版。

### 1.2 小目标一：车路独立感知与规划相关的空间 Query



#### 子目标 A：获取车端、路端独立 3D 对象观测

分别建立车端 D1、路端 D2 的独立对象观测接口，先统一来源、几何与时间合同，再选择检测前端。**首版从各端当帧原生标签建立 Oracle 输入；随后固定车端 GT 替换路端 D2，再固定路端协议替换车端 D1。**每个来源组合独立训练，固定权重换输入只作分布变化诊断。不得将 cooperative 合并 GT、协同融合后的预测或读取 GT 匹配结果的 query 当成任何一端的独立观测。

**已核查的检测路线（2026-10-04）：**UniV2X 与 UniMM-V2X 单端 stage-1 都是相机输入 → ResNet-101/DCN＋FPN → 时序 BEVFormer → 对象检测/跟踪 query → 3D 框，不向检测模型输入实测点云。两者均使用 200×200 BEV、256 维特征和十类分类头；UniV2X 为900个对象 queries、双端训练队列5帧；UniMM-V2X 为1500个对象 queries，车端4帧、路端5帧。本地 E5-A 训练采用双端3帧。训练队列长度不能代替实际推理历史记录。UniMM-V2X 单端配置的 BEV 层未启用 MoE，车端检测 decoder 显式 `is_moe=False`，不能把论文整体提升当作当前单端 detector 的已验证收益。详见[双库源码核查](independent_3d_observation_review_20261004.md)。

**Oracle 数据必须绕过 PKL 的跨帧补框：**车端读取 `vehicle-side/data_info.json` 的 `label_lidar_std_path`（`label/lidar`），路端读取 `infrastructure-side/data_info.json` 的同名字段（`label/virtuallidar`），保留原始类别、annotation token、track ID、遮挡/截断字段及实际标签版本。两库 `spd_to_uniad.py` 都会利用后续观测补出遮挡框、用下一帧计算 GT velocity；“单端 PKL”不能自动等同于当帧独立观测。本地路端 val PKL 有1,124/12,803个框的 annotation token 不在对应当帧原生标签中，涉及421/675帧；该问题已修复：`prepare.py` 仅使用PKL时间/位姿/前驱信息，直接读取当帧原生路端框，已导出`continuous_v3_native_gt`；不能仅靠禁用速度消除补框影响。该比例是标注来源差异，尚未量化其对 AP 或规划的影响。

**区分标注覆盖与相机可观测范围：**主 Oracle 组使用独立 LiDAR GT 的标注覆盖，明确为 Oracle perception；另设相机几何 FOV 限制的 Oracle 对照，以解释 GT→camera Detector 的退化。后者用本端相机内外参、原始图像边界与正深度判断框投影，并固定判据，不把 `pc_range` 或50m规划 ROI 当作相机可见范围。几何投影不证明目标未遮挡；原生遮挡标签可用于诊断，不能直接作为 Detector 的推理特征。相机时间戳与标签点云时间戳分别记录。当前 LiDAR GT 是 Oracle 对象来源，不代表给 VLM 输入点云；点云检测器若另做，应作为不同传感器 setting。

**Detector 首选复用现有 UniV2X 冻结权重与几何输出，UniMM-V2X 作为后续受控对照。**D1 单端图像推理本身不因评测使用 cooperative GT 就变成协同推理，但现有 D1 训练/评测标签协议不满足新版原生单端要求：旧车端 train PKL 曾为空文件、val仅20帧，现已备份并恢复完整1521train/675val；另建两端无补框Native版本。独立检测AP的nuScenes JSON evaluator仍需单独对齐/重评。先核查图像 BGR/归一化、标定、框尺寸/中心/yaw、scene 顺序与历史，再决定是否重训。官方权重不代表按新版独立标签训练；训练来源和预训练 scene 重叠须披露。现有675帧预测仅覆盖规划 test，不能据此完成 train/val Detector 组，需按合法过去顺序导出所需帧，不为复用缓存改变 split。

统一缓存分为宽候选观测和后续选择结果，至少记录 `agent_side`、`observation_source=gt/detector`、原始/统一类别、源坐标及当前 ego 坐标的 `xyz/lwh/yaw`、源 frame/scene、图像/标签时间戳、消息可用时间、标定/pose、score 与有效性、velocity 与有效性、可选 track/query 及其对应索引，以及 checkpoint/config/preprocessing/schema hash。共享编码器的 source embedding 区分 vehicle/RSU；GT/Detector身份留作审计，不作为“完美Oracle”提示。未知速度统一禁用并标无效，未知质量不得伪造为实测点数或协方差。只选截至 ego 时刻已可得的消息；无接收时间时明确无额外传输延迟的假设。原始观测和对齐版本均保留。最新修订：首轮先实现同track因果历史恒速补偿，缺失历史标无效；实现与结果见第3.2.1节。

检测候选 `boxes_3d_det/scores_3d_det/labels_3d_det` 与跟踪筛选 `boxes_3d/scores_3d/labels_3d/track_ids` 分开导出；首版采用 detection 候选＋validation冻结的阈值/同源去重/ROI/Top-K，tracking 另作消融。300候选不等于300对象，跟踪 query 特征也不能按检测框行号直接拼接。缓存生成必须走无标签匹配的推理路径；冻结检测器推理缓存可复用，训练态匹配的 query 不可冒充预测。当前结果不含视觉 query/BEV，需要时重新导出并保留 bbox 索引和 mask。

验收分别报告：每端来源/时间/类别/覆盖统计；独立标签协议下各类 AP、召回与漏检/误检分组；同输入生成后端下的规划 L2 与 coverage。保留已有十类 mAP=3.09%/12.19%；三类诊断平均=10.29%/40.62%不改称官方成绩。正式三类评测须版本化类别、范围、距离阈值、GT过滤与分母，不能为改分数直接缩小官方 checkpoint 分类头。Map Queries 需要真实地图标注、地图表示及 decoder；两库虽有地图 head，本项目对象接口尚未接入，首版 object-only，不以对象框冒充地图。

上节来源异常复查：路端额外框已用补框函数逐项重生成，1124/1124个val token及几何全部对应；这是官方离线监督/轨迹补全设计，不表示Detector推理使用未来。限制针对把补全GT直接作为VLM当前观测。车端原生数据实际足够生成1521train/675val；旧20帧val为应有列表的精确前缀，属于不完整本地产物。已恢复train保存，改为完整循环后原子发布；数据引用检查没有发现漏下载。历史中断原因未确定，已完成修复/重导/13项测试，详见[修复验收报告](independent_observation_repair_20261004.md)。

#### 子目标 B：对象处理、可变范围和时空对齐，生成 VLM 输入

```text
独立车端 GT / D1 ──────────────────────────────────────────┐
独立路端 GT / D2 → 消息时间审计 → 变换至当前 ego 坐标 ──────┤
                                                          ↓
                  宽候选集合 → 固定/速度/规划相关性选择 → 双端对象
                                                          ↓
                  路线 A：归一化 → 共享 ObjectEncoder + source → soft tokens
                  路线 B：对象位置 + 图像特征/3D位置或BEV → 空间/视觉 queries
                                                          ↓
                                                      VLM 输入
```

路侧相机无需看见自车；有效相对位姿和消息时间能确定路侧对象相对自车的位置。没有可靠位姿时，不能只凭像素计算自车距离。未知目标速度不做伪精确 TTC；来源/速度有效性/年龄显式保留。

分开设计宽候选范围、相关性选取范围和 token 预算。先固定范围 nearest-K，再速度自适应前向走廊，再预算内规划相关性排序。方向、横向潜在冲突、VRU、路端独有信息和年龄/不确定性逐步加入；停车时仍保留横向风险覆盖。所有选择只依赖截至当前可得信息，不读取未来 GT 轨迹。

已知标定做解析变换。可学习的时空适配只用于确实存在的语义域差异、合法历史运动补偿或有限残差，需有对应监督。用精确 GT + 精确变换做恒等重建，不作为创新 loss。时序 memory 可借鉴历史关联/补充，候选必须来自实际过去观测，跨 scene 重置；不存储按测试帧索引可检索的未来/测试标注。

**表示之间的关系：**13D 对象经 MLP 得到的是几何 soft token；BEV 是组织空间特征的中间表示；可学习 query 从图像/BEV memory 读取特征，再投影成 VLM token。这三者不是必须串起来的步骤，也不能把随机 MLP 输出直接称作包含视觉语义的 query。首版无需先构建 BEV。

### 1.3 小目标二：自车状态与因果历史的连续条件表示



#### 子目标 A：StateEncoder 编码当前数值状态和历史运动

将已审计的自车速度、加速度、尺寸和历史轨迹/有效 mask，按统一当前 ego 坐标与时间协议归一化，使用独立 StateEncoder 生成少量 state tokens。字段与目标框不同，独立于共享 ObjectEncoder；历史只使用当前及以前观测，审计速度/加速度是否由未来姿态差分产生。

StateEncoder 输出进入 VLM。通过 state-only、state 文本、state tokens 的受控对照验证数字信息是否被规划利用，不在主方案中给轨迹 head 添加数值状态旁路。

#### 子目标 B：历史记忆与通信鲁棒性

在稳定的 StateEncoder 基线上，研究带时间间隔/有效性 mask 的因果历史编码与可选自车 memory，评估路侧丢包、延迟时保持基础规划的能力。对象历史 memory 与自车历史分别管理；缺失 RSU 不删除自车状态/视觉。先比较无历史/短历史，再加入通信扰动，不能把使用历史本身直接等同于已获得鲁棒性。

最终两条输入和车端图像共同形成规划表示：

```text
车端图像 P ─────────────────────────────────┐
双端对象 soft tokens / 空间视觉 queries ────┤
自车状态/因果历史 → StateEncoder ──────────┤
固定驾驶提示 ───────────────────────────────┤
                                            ↓
                         VLM → 自回归生成轨迹坐标文本 → 解析/数值评测
```



### 1.4 参考代码、复用范围与开发边界

下列路径均为本地已存在的参考；“参考”不意味着已接入或可直接加载其完整权重。


| 任务                                   | 本地参考代码（相对所列项目根目录）                                                                                                                                                                                       | 复用范围与边界                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 车端/路端独立 3D 检测                        | DrivewithVLM：`projects/configs/Robodrivevlm/univ2x_e5a_vehicle_det.py`、`univ2x_e5a_infrastructure_det.py`；UniV2X：`projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py`、`dense_heads/track_head.py` | D1/D2 配置、独立推理输出、后续 query 导出；先核验真实 checkpoint/检测结果，不把已有配置当成检测已验收                   |
| UniMM-V2X 单端检测对照 | `projects/configs_e2e_unimmv2x/unimmv2x_sub_vehicle_stg1.py`、`unimmv2x_sub_inf_stg1.py`；`projects/mmdet3d_plugin/unimmv2x/modules/encoder.py`、`detectors/unimmv2x_track.py` | 同族BEVFormer前端；先核对实际MoE开关、query/训练配方和标签转换，不作为首版替换前置；详见独立3D核查报告 |
| 双端 GT 读取与缓存                          | DrivewithVLM：`tools/continuous/ego_perception.py`、`prepare.py`                                                                                                                                          | 原生车端 GT、独立 RSU、当前 ego 坐标、timestamps/masks；重导宽候选缓存，现有50m/车32+路16截断不能用于扩大范围实验       |
| 13D 对象表示                             | OmniV2X：`navsim/agents/omniv2x/core/infra_bbox_features.py` 的 `V2XObjectFeatureBuilder`                                                                                                                 | 数值合同、nearest-K/padding；将 NavSim Scene 改为 SPD，显式处理 lwh/wlh 与未知类别                   |
| 对象 encoder / source 编码               | OmniV2X：`navsim/agents/omniv2x/models/diffusion/dit.py` 的 `infra_bbox_encoder`、`_encode_infra_bbox_conditioning`                                                                                        | 轻量对象编码思想，重新投影到本地 D_lm；其主 planner 是 Rectified Flow/DiT，不是 VLM；不复用 planner 侧条件注入位置  |
| 车路空间变换与关联                            | UniV2X：`projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py` 的 `AgentQueryFusion`                                                                                                             | 标定约定、reference point 变换、合法关联/补充；不是现成 LLaVA encoder，独立时空适配 loss 需本项目定义             |
| 图像特征的 3D 位置                          | OmniDrive：`projects/mmdet3d_plugin/models/detectors/petr3d.py` 的 `position_embeding`                                                                                                                    | 相机标定、深度采样位置编码；不将坐标名 lidar 当成 LiDAR 点云输入，不宣称其直接实现图像→BEV                            |
| 视觉/对象/地图 carrier queries 与时序         | OmniDrive：`projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py`、`petr_head_map.py`                                                                                                            | 额外 carrier query 读取图像 memory、感知 query 交互、3D/时序位置；先对象后地图，随机 query 需要监督/适配训练        |
| queries 拼入语言模型                       | OmniDrive：`projects/mmdet3d_plugin/models/detectors/petr3d.py`、`models/dense_heads/llava_arch.py`                                                                                                       | 投影后 embedding 拼接/占位替换思想；其 wrapper 与本地 HF LLaVA 不同                                 |
| 当前 VLM 前注入和规划接口                      | DrivewithVLM：`tools/continuous/online_model.py`、`online.py`、`data.py`                                                                                                                                   | 已实现图像 merge 后 RSU soft-token/规划 embedding 注入；扩展双端共享 encoder 和 StateEncoder，保持在线梯度 |
| StateEncoder / 动态 selector           | 当前需新开发；输入来源参考 DrivewithVLM 的 `data.py`、`prepare.py`                                                                                                                                                     | 不是已有模块；独立、可开关、版本化，规则 selector 起步，再研究可学习选择                                         |
| semantic/spatial 融合、waypoint、VAE＋GRU | `docs/references/arxiv/03_2608.21032_AURORA.pdf` §4.1–4.3 / Appendix B                                                                                                                                  | 论文方法参考；无已确认可直接复用的完整本地实现，具体模板/先验后验细节不能自行写成原论文事实                                    |


项目根目录：DrivewithVLM=`/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM`；UniV2X=`/home/zzn/V2X_VLM/UniV2X`；UniMM-V2X=`/home/zzn/V2X_VLM/UniMM-V2X-main`；OmniV2X=`/home/zzn/V2X_VLM/OmniV2X-main`；OmniDrive=`/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main`。

原始参考：[OmniV2X 源码](https://github.com/JuntongPeng/OmniV2X)、[UniV2X 源码](https://github.com/AIR-THU/UniV2X)、[OmniDrive 源码](https://github.com/NVlabs/OmniDrive)、[AURORA 原文](https://arxiv.org/html/2608.21032v1)。动态空间选取已有相关研究：[Where2comm](https://arxiv.org/abs/2209.12836)、[面向驾驶的选择性通信](https://arxiv.org/abs/2305.17181)。可变距离本身暂不宣称新颖；研究假设是规划相关、时效感知、双端互补的有限 token 选择。

## 2. 所有融合方式共用的路侧接口



### 2.1 数据来源：Oracle 和 Detector 可互换

```text
路侧独立 GT boxes/tracks ── Oracle exporter ──┐
                                            ├─ 统一 Agent Cache ─ 对齐/编码 ─ VLM输入融合
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


| 字段                                     | 用途                  | 缺失时的规则                                |
| -------------------------------------- | ------------------- | ------------------------------------- |
| `score` / `score_valid`                | Detector confidence | Oracle 可设 1，但不能解释为现实检测器可靠性            |
| `age_s` / `timestamp`                  | 消息年龄与实际时间对齐         | 使用真实配对时间；不能默认所有路侧消息 age=0             |
| `source_id`                            | 区分路侧来源              | 单 RSU 可固定；与VLM输入的modality embedding区分 |
| `velocity_valid`                       | 速度是否可用              | 未知速度数值可填 0，但有效性必须为 false              |
| `state_valid`                          | 几何分量有效性             | 与整对象 padding mask 分开                  |
| `track_id`                             | 时间关联、后续 memory      | 作为关联元数据；不要默认将任意全局 ID 直接当可泛化 embedding |
| `uncertainty` / `uncertainty_valid`    | 质量估计                | 第一版可不启用；不能把 score 直接当 covariance      |
| `semantic_features` / `semantic_valid` | 后续检测 query 语义       | 第一轮 geometry-only 不启用                 |


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



## 3. 方案实施

2026-10-04最新输出协议：用户要求先保持VLM直接输出轨迹。本文首轮排期随之改为语言生成；第4节连续头代码不删除。已核实检测产物和split覆盖见[检测结果审计](d1_d2_existing_results_audit_20261004.md)。



### 3.1 技术方案与共同实验协议

**主方案：保留车端图像＋双端对象表示＋自车 state tokens → VLM 自回归生成轨迹坐标文本 → 固定解析器/数值评测。** 先保持已有语言输出合同，研究感知输入、范围及 queries；连续 MLP/VAE head 推迟，不作为首轮前置。已有 E2 checkpoint-932 有157条 validation 直接生成结果，但输入来源不同，需在新版独立双端协议下重建基线。


| 方案                            | 输入构造                                                                | 核心实验问题                                       |
| ----------------------------- | ------------------------------------------------------------------- | -------------------------------------------- |
| I1 原始视觉＋几何对象 tokens（优先）       | P + S_ego + S_rsu + S_state；共享 ObjectEncoder 加来源编码                  | 独立双端 GT 经 VLM 是否有价值？数值 soft tokens 是否优于对象文本？ |
| I2 OmniDrive 启发的视觉/空间 queries | 先 P + Q + S_state；Q 从视觉特征读取，并以对象位置/3D位置条件化；必要时保留未融合几何 tokens，明确重复输入 | 视觉语义能否补充仅几何 tokens？P+Q 主方案与 Q-only 压缩对照分开    |
| I3 AURORA 启发的跨源空间融合           | 车路空间表示对齐/关联/补充 → semantic/carrier 读取 → P + T_f + S_state            | 在保留 P 时，显式融合是否优于独立对象输入？T_f-only 替换视觉仅作消融     |


I2/I3 不是把 I1 的 MLP token 换名字。GT-only geometry 融合原型单独命名；没有真实图像语义/地图 memory 时，不宣称复现视觉 semantic/map queries。BEV 若启用是 I2/I3 的中间特征分支，先验证数据、视角和预训练可用性，再立专项里程碑。

VLM 序列采用 `[角色前缀] [图像/视觉块] [双端对象或融合块] [state/history块] [固定规划提示] [assistant生成前缀]`。所有观测在生成答案之前；序列轴拼接，统一 D_lm，正确维护 generation 的 attention mask、position 与 KV cache。直接生成不以 `<wp>` hidden 读出为前置；连续头相关要求保留第4节。禁止 head 拼接 RSU 向量、直接读 RSU memory 或用 RSU 在 VLM 后修正轨迹。融合后的对象不再次重复插入同一 RSU token 块。

当前 online_model.py 使用规划 soft embedding + 连续 head，尚不提供 soft-token 自回归生成。已有 `tools/robollm/test-ddp_token.py` 使用 `model.generate` 直接输出文本，但需新增双端 soft-token/StateEncoder 的训练与 generation 接口，不能直接复用只支持连续头的注入 hook。训练新投影/StateEncoder 时，必须保留通过语言网络的梯度； detached hidden 缓存无法支持这些模块或 LoRA 联合训练。先新增模块预热，若接口可拟合但泛化不足，再统一比较 LoRA 联合适配；E2 adapter 与 base 初始化对照不得只给某一种方案额外预训练。

首轮固定9步、0.5s的轨迹文本输出合同，使用相同数值格式、生成参数、解析器和数值 evaluator。训练用 assistant 答案的语言交叉熵，观测 tokens 标签屏蔽；连续 waypoint loss 不作为首轮主损失。同生成后端、来源、split、prompt 信息、训练预算独立训练。对象 soft tokens 主组去掉重复对象文本；状态文本改 state tokens 单独比较。GT future 仅作标签，消息选择/速度/路径走廊不读取未来答案。Oracle 双端信息不等于完美同步信息。

选择规则：宽候选缓存 → ego 相关区域 → 固定预算 K。速度规则可从 `R(v)=clip(Rmin+|v|Tlook+margin,Rmin,Rmax)` 起步，实际优先前向走廊＋基础横向覆盖。近场/VRU/路端独有对象配额、年龄/不确定性、跨源匹配逐项消融。高分辨率 BEV/类别峰值/NMS 属于后续 Detector 优化，去重半径与 ego ROI 不混用；TF-ETCP 同名原始来源未核实，仅参考用户描述的因果历史关联/补充思想。

#### 3.1.1 实验方案与首轮结果（2026-10-05）

I1/I2/I3是输入方案编号，T1/S1是首轮实验编号；**S1即I1的当前实现**，不重复计为两个模型。当前T1/S1的自车状态和因果历史均为文本，StateEncoder尚未接入；下表区分已完成实现与后续目标。

| 方案 / 实验 | 原始车端图像 P | 双端对象 / 空间输入 | 自车状态与因果历史 | 输出与训练 | 当前状态 / 实验目的 |
|---|---|---|---|---|---|
| T1 对象文本对照 | 保留 | 同一批独立双端GT，经因果补偿、A1关联、50m范围和固定预算后转文本 | 文本 | 语言LoRA；VLM直接生成9步轨迹文本 | 已完成6epoch及480帧test；作为数值soft tokens的同信息对照 |
| I1 / S1 几何对象soft tokens | 保留 | 与T1完全相同对象；共享ObjectEncoder＋xyz位置编码＋来源/质量编码，每对象一个token | 当前为文本；StateEncoder另做消融 | 语言LoRA＋ObjectEncoder；VLM直接生成9步轨迹文本 | 已完成6epoch及480帧test；比较表示精度和输入开销 |
| I2 视觉 / 空间queries | 主方案保留 | 由视觉特征读取Q，以对象位置 / 3D位置条件化；几何与视觉重复输入显式记录 | 首轮与T1/S1相同文本；state tokens独立比较 | 保持同一VLM生成后端；新增query模块独立训练，预训练来源披露 | 待实现；比较P＋Q与I1；Q-only另作压缩消融 |
| I3 跨源空间融合queries | 主方案保留 | 车路空间表示对齐、关联、补充，再读取semantic/carrier tokens T_f；不把GT几何MLP称为视觉semantic融合 | 同I2；不同时更换状态接口 | 保持同一VLM生成后端；新增融合模块独立训练，预算披露 | 待实现；与固定carrier的I2比较显式融合；T_f-only另作消融 |

共同协议：原生独立双端GT Oracle，`dual_object_a1_v1_20261004`，scene split为train929 / val157 / test480；坐标为当前ego LiDAR、因果历史运动补偿、A1关联、50m ROI、32个ego贡献＋16个额外RSU对象。T1/S1使用同一缓存，物体速度不作为输入；原始车端图像与state/history文本一致。相同基础LLaVA，seed42、6epoch、1398次更新、有效batch4，语言LoRA r/alpha=8/8，视觉编码器/projector冻结。validation按coverage优先、平均L2次之选checkpoint；test仅报告，不重新选epoch。所有路侧信息先进入VLM，无planner旁路。

首轮实测（test480；误差单位m）：

| 方案 / 实验 | val选中epoch | L2@1s ↓ | L2@2s ↓ | L2@3s ↓ | 平均L2@1/2/3s ↓ | FDE@4.5s ↓ | 解析成功 | 平均输入tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 对象文本 | 5 | 0.1099 | 0.5764 | 1.5095 | 0.7319 | 3.8247 | 480/480 | 3076.5 |
| I1 / S1 对象soft tokens | 6 | 0.1030 | 0.5642 | 1.4802 | 0.7158 | 3.7948 | 480/480 | 1198.4 |
| I2 视觉 / 空间queries | — | — | — | — | — | — | — | — |
| I3 跨源空间融合queries | — | — | — | — | — | — | — | — |

S1主平均L2较T1降低2.20%（绝对0.01613m），输入tokens减少61.05%；254/480帧及12/19场景改善。单seed结果，不能宣称显著或稳定胜出，也不能证明路侧相对ego-only的收益。后续需同构ego-only对照和多seed复验。I2/I3未完成，表内不填预期成绩。

原始证据：[完整test报告](../evaluation_results/T1_S1_test_20261005_114004/report.md)、[逐帧配对分析](../evaluation_results/T1_S1_test_20261005_114004/paired_analysis.json)。论文总表见[实验计划第2.2节](v2x_vlm_model_optimization_and_experiment_plan.md#22-论文结果总表模板)。

**长时域baseline对照（2026-10-05）：**采用用户提供的2.5/3.5/4.5s表头。V2X-VLM行是用户提供的参考值，尚未核实来源、split、样本数、输入、GT与collision checker；T1/S1行来自本次480帧test逐帧预测重算。当前仅并列展示，不视为统一协议下的公平排名。原论文/外部报告结果与本项目实测来源明确区分。

| Method / Settings | L2 Error (m) ↓ 2.5s | L2 Error (m) ↓ 3.5s | L2 Error (m) ↓ 4.5s | L2 Error (m) ↓ Avg. | Collision Rate (%) ↓ 2.5s | Collision Rate (%) ↓ 3.5s | Collision Rate (%) ↓ 4.5s | Collision Rate (%) ↓ Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2X-VLM (Baseline；用户提供，协议待核实) | 1.09 | 1.12 | 1.42 | 1.21 | 0.02 | 0.03 | 0.03 | 0.03 |
| T1 双端GT对象文本；本项目test480 | 0.9740 | 2.1524 | 3.8247 | 2.3170 | — | — | — | — |
| I1 / S1 双端GT对象soft tokens；本项目test480 | 0.9564 | 2.1246 | 3.7948 | 2.2920 | — | — | — | — |
| I2 视觉 / 空间queries；待实验 | — | — | — | — | — | — | — | — |
| I3 跨源空间融合queries；待实验 | — | — | — | — | — | — | — | — |

此表Avg.为三个终点时刻L2的算术平均，**不是0.5–4.5s全部九点平均，也不是此前1/2/3s平均**。Baseline按用户给值保留（可能存在四舍五入），碰撞字段单位为百分比；本项目碰撞未评测，不能填0或移用baseline值。T1/S1在本次协议下的长时域平均分别2.3170/2.2920m，S1降低约1.08%；2.5s低于所给baseline，3.5/4.5s高于所给baseline，但协议未统一，不能由此下公平胜负结论。

补充4s指标：T1 L2@4s=2.9286m，S1=2.9043m。数据证据：[长时域指标](../evaluation_results/T1_S1_test_20261005_114004/long_horizon_metrics.json)。无需重新推理，使用相同480帧预测与原始GT，索引4/6/8分别对应2.5/3.5/4.5s，索引7对应4s。

#### 3.1.2 I2开发依据与边界（2026-10-05）

用户将状态／长时域轨迹部分另行并行开发，本分支专注VLM前的视觉／空间queries。已核查本地OmniDrive、ORION、VGGDrive及原文，详见[I2代码核查与开发方案](i2_visual_spatial_queries_code_audit_20261005.md)。

- OmniDrive：本地Llama系LLaVA-7B＋EVA-02＋Q-Former-PETR，以专用carrier读取带3D位置编码的图像memory；不是13D对象MLP直接改名。
- ORION：Vicuna v1.5＋EVA-02，QT-Former增加因果历史memory读取；其主规划器不纳入当前I2。
- VGGDrive：Qwen2.5-VL-7B＋冻结VGGT，保留视觉tokens，并在decoder层内用视觉hidden读取几何memory；与输入级P＋Q是不同分支。
- 首版I2-A保持当前SigLIP／LLaVA与同一对象合同；对象条件query读取车端patch memory，保留P，用Q替换S，state/history仍文本。无法取得对象局部视觉支持时保留几何，不给RSU-only对象编造车端图像语义。
- 明确区分对象xyz／2D投影原型与真正PETR射线3D patch位置编码；首版不建立BEV。输入级I2-A先验收，再研究I2-B路端视觉queries及VGGT扩展。
- 增加路端图像语义属于额外输入来源，须设置同双端视觉对照；不与几何-only I1混称严格同输入。保留P＋Q的视觉重复信息和预算明确披露。
- 新代码隔离在tools/object_vlm_i2，先标定／mask／query／KV smoke，再用户启动训练；旧T1/S1代码hash、权重和cache不改。当前只完成核查与接口设计，I2尚未实现或运行。

### 3.2 实验排期与依赖

排期从下一轮实际开发开始算，按工作日估计。**主线初步窗口 4～6 周，不是已测算的 GPU 完成承诺。** 前2天先测真实7B吞吐，再按可用 GPU、训练步数和实现问题调整。检测器重训、LiDAR/BEV新栈及地图建设不包含在该窗口内。未完成前置验收时顺延，不能为赶日期跳过检查。


| 顺序 / 初步窗口                   | 工作与实验                                                          | 交付 / 进入下一阶段的条件                                         |
| --------------------------- | -------------------------------------------------------------- | ------------------------------------------------------ |
| P0：第1周前半，约2–3天              | 独立 GT/位姿/时间/自车数值审计；重导宽候选；核验 D1/D2 现有权重状态                       | 新 cache/manifest、可用字段/覆盖统计、无未来输入；真实7B前向/反传/显存吞吐结果      |
| P1：第1周后半至第2周，约4–6天          | 双端共享 ObjectEncoder/source、独立 StateEncoder、有效性与在线保存恢复；A组接口/状态对照 | CPU必要检查＋真实模型小样本验证；对象/state有梯度；观测/答案隔离及generation前缀正确             |
| P2：第2–3周，约4–6天＋训练           | I1固定范围的 B0/B1/B2、必要的对象文本对照；冻结新增模块预热，按诊断决定 LoRA 适配              | 完整validation、state-only/恒速基线、ego/RSU增益与失败案例；不能只报训练loss |
| P3：第3–4周，约4–6天＋训练           | C组固定/速度/规划相关性选择；同预算、同最大覆盖                                      | 选择日志、预算效率、关键对象漏选诊断；验证是否只是多看对象带来的收益                     |
| P4：第4–6周，约5–8天＋训练           | I2和I3对象优先原型与对齐预热；D组视觉/融合对照                                     | 三接口同来源初筛；map/BEV依赖缺失则明确缩小原型，另列扩展排期                     |
| P5：依赖D1/D2真实输出，追加约3–5天适配＋训练 | E组先RSU Detector、再ego Detector；因果历史memory/丢包延迟分开                | 真实预测合同、检测指标与规划退化/收益；不拿GT冒充Detector                     |
| P6：接口选定后另排                  | VAE＋GRU vs 同输入MLP；候选方法多seed复验、最终test                           | 当前直接生成基线保留；后续decoder收益独立归因；test用于最终报告                        |


P0/P1可以开发新版接口，不受旧M1 F0失败阻塞。新版若数值状态与生成行为仍不可用，先修复再解读协同价值；不直接铺开九组完整训练。所有模型/数据用新版本目录，保留旧缓存和checkpoint。

#### 3.2.1 首轮缩减：仅 T1 与 S1（最新用户决定）

**最新GPU实测：**T1/S1真实7B单卡及四卡短训练均已通过，34项CPU相关测试通过。正式训练为T1先使用0–3四卡6epoch，再S1同四卡6epoch，用户自行启动；短probe轨迹解析0/8，只证明工程接口可运行。见[smoke报告](../work_dirs/T1_S1_objects_4gpu_smoke_20261004/report.md)。

**T1/S1最新代码状态（2026-10-04）：**共同A1对象缓存已导出929/157/480，`tools/object_vlm/`已实现文本/soft-token自回归训练与generate，31项CPU相关测试通过，统一长度7168。真实7Bsmoke尚未通过（显存被其他任务占用），正式训练未启动；用户自行用`bash tools/object_vlm/run_t1_s1.sh`运行。详见[实施记录](t1_s1_object_vlm_implementation_20261004.md)。此前“输入导出与生成接口未完成”的描述为本次实现之前的历史状态。

**关联放宽V2已完成：**完整train929/val157；L1取消距离减半，L2进一步对≤100ms且几何证据较强的未知运动消息允许配对，三组统一原始尺寸。val召回79.02%/80.65%，错配均1条，仍低于A1的88.46%，故默认继续A1。21项关联测试通过，未使用test、未接入VLM。详见[新对照报告](../data/Planning/relaxed_fusion_v2_trainval_20261004/report.md)。

**最新实施进展（2026-10-04）：**双端因果尺寸稳定、路端运动补偿、中心距离/朝向/软IoU关联、高可信判断和尺寸选择/平均已实现，A0–A4完成train929/val157及冻结后test480，28项相关测试通过。详见[实施计划与状态](causal_size_stabilization_fusion_plan_20261004.md)和[cor验证报告](../evaluation_results/stable_fusion_v1_20261004/report.md)。补偿提高val召回67.60%→88.46%；尺寸稳定降低波动，但A3最终召回63.87%，A4未改善cor尺寸一致性，因此冻结保留A1。cor尺寸继承ego的偏向不能作为真实物理精度证据。T1/S1共享输入导出、ROI/top-K和VLM生成训练仍未实施；原生GT未改动。

**前置任务最新进展（2026-10-04）：以下三项已实现并完成完整 train/val CPU 实验。**

1. 建立 cooperative 参考匹配对：两端原生 frame/sequence/timestamp/track/token 回查，唯一且一致才有效；缺失或冲突为未知。全SPD确认10,567条双端对应；仅作离线评估，不进入匹配器或VLM对象输入。
2. 因果运动补偿：原生路端当前及最近过去同sequence/track，历史间隔不超过1s；世界坐标估计平面绝对速度、外推到ego时刻再变换。无历史保留空间对齐框，motion_valid=false。历史PKL速度/未来补框禁用。
3. 组合关联对照：A=BEV IoU，B=BEV IoU+距离+尺寸，C=B改用upright 3D IoU；共同类别/距离/IoU门限、Hungarian一对一并允许未匹配。两种时序处理及108组参数全部扫描，包含类别/年龄分组。尚未锁定正式去重参数。

入口 `python -m tools.association.run --output-dir <不存在的新目录>`；实现说明见[关联模块README](../tools/association/README.md)，实测与产物见[关联实验报告](native_association_motion_audit_20261004.md)。7项测试通过。主产物 `data/Planning/association_native_v1_final_20261004/`。只运行train929/val157，未做test调参、未训练VLM。

默认诊断门限车辆2m/行人骑行者1m、IoU0.1下，val参考召回率67.60%→88.46%；车辆参考对中心差1.87→0.93m。B本设置未胜过A，不宣称组合代价必然更优。精确率只覆盖可判定预测，未知匹配另报。补偿先于ROI/top-K；T1/S1后续必须共用相同补偿和选择结果。现有旧planner/训练缓存未自动升级，旧代码状态继续按历史记录理解。


本轮只做两组，暂不训练ego-only、四组接口对照、可变范围、queries或连续head。P0数据/接口检查仍必做；其余第3.2表格为后续排期，不是本轮需执行的清单。

| ID | 车端对象 | 路端对象 | 表示 / 输出 |
|---|---|---|---|---|
| T1 | 独立vehicle-side GT | 独立infrastructure-side GT | 同字段对象文本 → VLM直接生成轨迹 |
| S1 | 与T1完全相同的对象 | 与T1完全相同的对象 | 共享ObjectEncoder + ego/RSU来源编码的soft tokens → VLM直接生成轨迹 |

两组固定车端原始图像；首版不额外提供RSU原图，保持结构化路侧输入假设。固定50m、ego最多32/RSU最多16、相同消息配对/标定/年龄、相同缺失速度策略，文本描述完整同信息字段；S1删除重复对象文本。自车state/history两组采用同一表示：最小实验先保留因果文本，StateEncoder单独延期；若后续先实现StateEncoder，则两组同时启用，不仅给S1启用。最小两组验证的是对象表示，不称完整P+S_ego+S_rsu+S_state已全部接入。

共用train/val/test split、基础VLM、LoRA配置、训练步数、trajectory输出模板和解析器。新增对象encoder的预热计入预算并披露；对照双方从相同语言初始化独立训练，不拿旧双图像模型直接当T1。先小样本验证在线梯度、generate/KV cache和保存恢复，再完整validation。未来GT只作语言标签，观测tokens不参与答案CE。

两组可回答“相同双端GT信息下soft tokens是否优于对象文本”，**不能回答路側相对ego-only有多少增益**；需要时再补T0/S0。报告L2@1/2/3s、主平均、4.5sFDE、解析失败/coverage与token数/时延。单seed先探索，候选结论再复验。

当前运行的 `bash tools/robollm/dist_train.sh projects/configs/Robodrivevlm/MMdrive_v2x.py 4` 归为旧双图像+state/history文本基线，不计T1。2026-10-04宿主机进程确认其4-rank训练，output=`checkpoints/V2X_dt_9step_4096`，6epoch、LoRA r/alpha=8/8、lr=2e-4、seed42、eval_strategy=no；已保存step368(epoch1)和736(epoch2)。训练JSON1470条/测试JSON676条均无Perception对象段，图像数2。当前pipeline未独立读取两端GT对象并转为文本；实际读JSON的QA_pairs，文件中存在未调用的对象formatter不代表已接入。保留运行，不修改其config/数据或抢占设备。详情见[现运行身份核查](running_v2x_t1_eligibility_20261004.md)。

#### 3.2.2 后续：视觉/空间 Queries、真实检测与鲁棒性


| 组别       | 实验                                                 | 归因要求                                               |
| -------- | -------------------------------------------------- | -------------------------------------------------- |
| D1       | I1 vs I2：P+几何tokens vs P+视觉/空间queries；再做Q-only压缩消融 | 共享GT/选择/state/LLM与输出合同；额外视觉预训练、query数和重复几何信息明确披露   |
| D2       | I2固定carrier，开/关车路显式对齐关联融合，形成I3桥接                   | 先固定解析坐标变换；学习适配/关联分别开关。路端独有对象保留，GT几何原型与真正semantic分开 |
| D3（扩展）   | 图像3D位置 vs 真正BEV memory；object-only vs object+map   | 有对应标定/特征/地图数据后开展，全部对照同等可得信息；新栈单独排期                 |
| D4（开题对照） | 同一批对象的soft tokens vs 投影到车端底图的几何视觉提示                | 不把框提示称为纹理重构；同源、同预算，车端视锥外对象如何保留需明确                  |
| E1       | ego GT固定，RSU GT vs冻结D2预测（F2）                       | 分别独立训练；固定权重换来源另列分布变化诊断                             |
| E2       | RSU协议固定，再换ego D1；最终双Detector                       | 不同时替换两端后直接归因某一端                                    |
| R1       | StateEncoder无历史/短历史/因果memory                       | 固定当前状态；有效时间间隔、历史长度和scene reset明确                   |
| R2       | 独立控制RSU丢包、延迟、标定噪声；对象历史补充开/关                        | 真实扰动与模拟分别标明，禁止未来观测补洞；GT当前框与Detector漏检收益分别解释        |
| T1（末期）   | 选定输入接口的直接生成 vs MLP vs条件VAE＋GRU                             | 连续planner只读取VLM表示；定义训练后验/推理先验、KL与采样协议，不能推理读取GT未来     |


时空loss以合法跨时刻对应/实际语义特征为监督；没有对应数据时保留解析基线，不硬加形式化loss。Detector小目标BEV分辨率/峰值/NMS优化另列检测实验，先证明AP提升再测规划，不移植用户给出的NMS半径作为对象范围。

新方法初筛统一训练配方/seed；候选结论建议3个seed，按scene统计差异。记录样本、优化步数、预训练、显存和有效参数。最终test在选型后报告。车端收到完整RSU消息才筛选仅减少VLM计算；若要证明通信带宽改善，选择要部署在发送端或请求协议中，计入ego状态反馈与时效成本，并实测字节数（坐标、语义维度、精度、协议开销）。

## 4. 保留的系统、代码与实验记录

本节保留已实现行为、历史网络代码说明和实验事实。**其中旧M1的D+256 planner拼接、离线hidden训练、旧文本输入及旧“下一步”描述仅供追溯，不能覆盖第1、3节。**旧oracle/detector planner旁路已禁用；保留代码说明不表示恢复该路径。旧M1～M8方案枚举、旧实验路线和未创建的文件建议已从本文删除，实际代码/缓存/权重未删除。

### 4.1 当前在线 I1 实现状态（2026-10-04）

已新增`tools/continuous/online_model.py`和`online.py`。I1保留真实图像merge，将路侧soft tokens与专用planning soft embedding注入语言decoder之前，trajectory head只读规划hidden。旧缓存train/model的oracle/detector planner旁路已禁用；旧F0仍可重载。具体命令和限制见[连续入口README](../tools/continuous/README.md)。

15项CPU检查通过，包括本地同族Qwen2/SigLIP的tiny HF模型、image/RSU影响、padding/empty、梯度和save/load。真实7B前向/训练尚未执行。当前只实现路侧输入核心；车端对象/state仍为文本，新增规划参数为soft embedding而非注册token；语言LoRA联合训练、时间补偿、I2/I3和VAE尚未实现。双端共享编码器、StateEncoder 和动态选取尚未接入。

### 4.2 HF LLaVA 接口适配清单（保留的系统要求）

1. 从实际backbone输出获取P和现有projector，保留图像展开、crop/patch配置；I1不改变视觉基座身份。
2. I2/I3在语言forward之前生成Q/T_f。参考OmniDrive的carrier机制，但先核验图像backbone、标定、map/temporal memory、预训练权重兼容性；随机query压缩不是零成本替换。
3. 建立统一`assemble_observation_embeddings`：输入各块embedding与valid mask，输出`inputs_embeds`、merged attention mask、position IDs、labels和显式wp_index。依本地4.45.2实现复用/改造图像merge；避免同时给pixel_values和已合并embedding重复插图。
4. padding对象从有效上下文剔除或按mask严格屏蔽；全空对象不得做全mask softmax。F0不加入RSU对象块，保留同一模块容量/初始化规则。有效空消息与丢包状态在数据层区分；若加状态token需独立消融。
5. 须是明确token/独立可训练embedding，保存tokenizer与该参数；定位图像展开后的index，不能读取模板末尾换行。未来轨迹、future动作/CoT只能作为标签；观测块labels为-100，文本辅助答案才有语言标签。
6. 注入spatial/state/carrier投影需要通过语言计算图得到梯度；冻结基座不等于整段forward加no_grad。离线h缓存不能训练输入投影、task token或语言LoRA。可缓存冻结视觉memory与冻结D2输出；在线经过trainable接口及语言网络。
7. 新checkpoint保存projector/encoder/carrier/aligner/任务embedding/LoRA/head/归一化及base身份；已有旧缓存拒绝跨输入版本复用。



### 4.3 旧 M1 网络与训练代码说明（仅保留追溯）



#### 4.3.1 第一版锁定的同构网络

```text
Ego image + causal history/state
               ↓
Frozen VLM；observation-only forward
               ↓
h_ego [B,D] ─────────────────────────────────┐
                                             │
当前配对 RSU objects → 当前 ego 坐标变换       │
               ↓                             │
Normalize [B,16,13]                          │
               ↓                             │
Shared Agent MLP：13 → 256 → GELU → 256        │
               ↓                             │
Masked mean → h_rsu [B,256]                  │
               │                             │
               └──────── concat ────────────┘
                              ↓
                相同结构 trajectory MLP [D+256 → 18]
                              ↓
                         [B,9,2] Δxy
                              ↓
                         cumsum → xy
                              ↓
                    唯一 masked Smooth-L1 loss
```

F0/F1/F2 使用同一 model class，配置仅切换 `rsu_mode=zero/oracle/detector`。F0 的 `h_rsu=zeros(B,256)`，不能改为 D 维输入的另一套 MLP。保留同一 AgentEncoder，保证总结构和总参数量一致；F0 路侧分支没有有效梯度，因此需区分总参数量与有效更新参数量，不宣称优化过程完全相同。

```python
# 同一模型、同一 D+256 输入 head；不同实验独立初始化并训练。
if rsu_mode == "zero":
    h_rsu = h_ego.new_zeros((h_ego.shape[0], 256))
else:
    # 标定/归一化在 exporter 或统一 transform 中完成。
    # 有效对象必须有限；padding 在 MLP 前后均清零。
    safe_geometry = torch.where(agent_mask[..., None], geometry, 0.0)
    agent_tokens = agent_encoder(safe_geometry)
    agent_tokens = torch.where(agent_mask[..., None], agent_tokens, 0.0)
    count = agent_mask.sum(1, keepdim=True).clamp_min(1)
    h_rsu = agent_tokens.sum(1) / count
fused = torch.cat([h_ego, h_rsu], dim=-1)
delta_xy = trajectory_head(fused).reshape(-1, 9, 2)
```

各实验由相同预训练 backbone、相同 seed 的 head/encoder 初始化、相同冻结策略、split、输入长度和训练预算独立训练。这里“从头训练”指新规划分支独立训练，不是重新预训练 VLM，也不是把训练好的 F1 关闭 RSU 当成 F0。主比较不从已训练 F0 继承 head；如后续采用 warm-start，单列为另一协议。

首轮只保留当前配对时刻对象状态和对象 mask；不加入 agent history、future motion、uncertainty embedding、source embedding、scene graph、CoT、cross-attention、Flow、risk Top-K、RSU dropout 或辅助语言/action loss。时间、来源、速度有效性仍保存为审计元数据，不作为额外学习特征。

坐标变换与归一化是必做的数据正确性工作，不是可省略的架构消融。RSU→world→当前 ego 的显式变换可用已审计的等价标定链实现。第一轮 nearest K=16、固定 ROI、类别映射及归一化规则在 Oracle/Detector 间一致。位置、尺寸、速度按固定物理尺度或训练集统计归一化；sin/cos 和 one-hot 保留。缺失速度的处理在启训前固定：补充可得的因果估计，或为 Oracle/Detector 统一采用禁用速度的独立 setting；不要把未知速度与实际静止混同而不披露。

#### 4.3.2 h_ego 与 observation-only forward：首要前置审计

冻结 VLM 最后一个 prompt token 的 hidden state 并不天然是适合连续回归的视觉表示，可能主要编码提示语言。因此在启用 RSU 前，必须先确定当前 LLaVA 的实际 readout 路径：

1. 核对本地模型/transformers 实现，记录 `h_ego` 来自哪一个语言 decoder 层，是否经过 final norm，以及返回 tensor 的真实序列长度。第一版候选为最后 decoder 层的最终上下文表示，不把返回索引约定当成已验证接口。
2. 输入仅包含车端图像、因果 history/state 和统一固定规划提示；删除现有 GT perception 文本、RSU 图像、assistant GT trajectory/action。future targets 独立放入 batch 字段，绝不进入 VLM 输入。
3. 取图像 token 合并/展开后的最后一个有效上下文 token，而非 padding/EOS 后的位置；readout index 必须与该 hidden sequence 对齐。左右 padding、batch>1、不同图片 token 数都要检查，不能无条件使用 `attention_mask.sum()-1`。
4. 固定 prompt 模板、backbone checkpoint、layer/token/readout 方法，F0/F1/F2 共用。第一版不新增随机初始化 `[PLAN]` embedding，也不把一个未训练 PLAN token 当作冻结模型天然支持的 readout。
5. Frozen VLM 使用 eval 模式关闭 dropout；同构模型 train 时保持 backbone eval。无可训练输入 adapter 的第一版可用 no_grad 提取特征，MLP 正常反传。以后若需要 LoRA/soft-token 则重新开放相应梯度。
6. 可离线缓存 frozen h_ego 加速比较，但缓存记录 model/template/readout/preprocessing/split hash，且两个 RSU setting 共用同一 ego feature；启用视觉 augmentation 时明确缓存协议限制。

若 F0 学不起来，先修 forward、token 位置、图像输入和目标对齐；再单独试多 token pooling、视觉特征 readout 或 LoRA。readout 改变后必须重新建立同构 F0 与 F1，不能仅给 Oracle 分支换更强表示。

#### 4.3.3 首轮只用累计 waypoint 的 masked Smooth-L1

第一版保持 T=9、dt=0.5 s；主报告提取 1/2/3 s，4.5 s FDE 单列。定义 p_0=(0,0)，Δp_t=p_t-p_{t-1}，预测累计轨迹为 p_hat_t=sum_{i≤t} Δp_hat_i。

**Head 输出 displacement，但唯一监督施加在 cumsum 后的 waypoint 上。** 不加末点额外权重、displacement loss、velocity/acceleration/collision/direction/smoothness loss、Flow loss或 action/语言辅助 loss。

```python
delta_xy = trajectory_head(torch.cat([h_ego, h_rsu], dim=-1))
pred_xy = delta_xy.reshape(B, 9, 2).cumsum(dim=1)
# 先屏蔽/清理无效 target，避免 NaN * 0；有效 target 必须有限。
safe_gt = torch.where(future_mask[..., None], gt_xy, pred_xy.detach())
point_loss = smooth_l1_loss(pred_xy, safe_gt, reduction="none").mean(-1)
loss = (point_loss * future_mask).sum() / future_mask.sum().clamp_min(1)
```

这里每个 waypoint 的 x/y loss 取均值，再按有效 waypoint 数归一化；固定 Smooth-L1 beta 和计算单位并写入 config。有效点监督使用未四舍五入的米制 GT。全无有效 future 的样本在数据准备阶段剔除，clamp 只用于防护；不能将这类样本当正常训练样本。DDP 下若各 rank 有效点数不同，应明确全局 valid-point 归一化，不能无意将不同 rank 的局部均值等权当作该全局公式。

缺失 future 的填零只作存储 padding，不作有效监督。Displacement 本身不修复 GT/插值问题，也不保证运动学可行性；仍需检查 timestamp、mask 和预测速度的诊断图。

### 4.4 首版实现与实验记录（2026-09-30）

F0/F1 的 M1 工程入口已落地在 [tools/continuous](../tools/continuous/README.md)。
复用现有 ContinuousTrajectoryHead，独立冻结并缓存 ego VLM 特征，再训练
AgentEncoder 与规划头；未在旧文本 Trainer 中混入新的监督。此方式与端到端
冻结 VLM 的计算图一致，前提是使用固定 eval backbone、固定观测和无图像增强。

- **同构模型**：zero/oracle/detector 保留相同 encoder、D+256 head 和初始化；
F0 强制 h_rsu=0。输出 9×2 displacement，经 cumsum 后仅作 valid-mask Smooth-L1。
- **readout**：本地 LLaVA/Qwen decoder 的 final-normalized hidden，按图像实际
合并后的 attention mask 取最后有效 token。缓存路径只输入车端图像、因果速度
和历史，不输入 assistant 回答、未来 GT 或路侧对象。
- **数据**：新增 continuous_v1 的 train/val/test=929/157/480，保留原 token split，
重建未四舍五入标签。独立路侧 GT 经显式坐标变换、尺寸/heading 解码、50m ROI
和最近 K=16 筛选；padding 与零对象显式处理。
- **因果限制**：首版所有对象速度置零，避免读取可能依赖未来帧的 GT velocity。
选 ego 时刻前已经可用的路侧消息；部分消息年龄约 2.2s，尚无运动补偿。
age/source 仅审计元数据，不是首版 encoder 输入，不能称为同步完美 Oracle。
- **验证**：17 项相关测试通过，包括实际 tiny LLaVA image merge、batch padding
readout、mask/空对象、坐标/yaw、两样本反传和保存加载。

真实本地 7B VLM 已在 CPU BF16 上提取 2 条 train 与 2 条 val 特征；F0 用
缩小的 head_dim=64 运行 30 epoch，train loss 从 10.08 降至 0.36。保存加载
预测一致，现有 evaluator 与训练评测一致。该极小 smoke 的最佳 Avg L2@1/2/3s
为 1.736m，恒速为 0.554m，**未达到正式 F0 门槛**；不能由此判断泛化能力或
路侧增益。记录保存在 `work_dirs/continuous_F0_smoke_20260930/`。

下一步是完整 train/val 的共享 ego 特征缓存与默认 head_dim=512 的正式 F0，
检查基线、预测方差与 9 点可视化；通过后才独立训练 F1。检查时所有 GPU 均有
正在运行的训练，本次未抢占设备或启动 F1。F2 仅预留统一输入合同，D2 native
prediction→13D exporter、跨对象 attention 与 Flow 均属后续工作。

### 4.5 旧车端文本输入的系统改造记录

用户明确最终模型的车端 prompt 包含周围 3D 对象与驾驶提示。正式 F0/F1 应以
此为主协议，而不是只用图像、自车速度和历史。旧 V2 的 `build_question()` 已有
`Perception and Prediction`、自车尺寸/速度/加速度/历史；旧 system template
还有规划角色、坐标方向和 4.5s/9点定义。当前 continuous-v1 未保留对象文本，
因此已完成的小样本运行只算简化输入工程 smoke，不能代替正式 F0。

目标输入为：车端图像 + **车端独立感知结果的文本** + 自车状态/历史 + 统一
规划提示 → 冻结 VLM → h_ego。F0/F1/F2 使用完全相同的该输入与同构规划头，
只切换 zero/Oracle RSU/Detector RSU 分支。Ego-only 表示没有路侧输入，不表示
车端只有图像。路侧对象不得先放进这份车端文本，否则 F0 已包含协同信息。

现有 `univ2x_e5a_vehicle_det.py` 提供 D1 车端独立 detector 配置，可从冻结的
D1 输出缓存构建车端对象文本；检测器位于 VLM 上游，不必给 VLM 新增检测头。
格式需明确类别、位置、尺寸、heading、有效速度及置信度、ego 坐标、ROI/Top-K
和缺失策略。无可靠因果速度时显式标注 unknown，不能伪装成可信零速度。

旧 `perception_text(info)` 读取 `instances` 标注而非 detector prediction，
且其输入来自 cooperative PKL，不能直接整体搬进 F0 当作车端观测。先审计
对象来源与可观测范围；若建立车端 Oracle 调试组，必须独立标识，并在所有
路侧对照中固定相同车端来源。旧对象文本实际上主要包含类别、xy 与速度，
不能称为已提供完整 3D box。固定驾驶提示可以进入模型，GT future trajectory
及由其产生的 action/CoT 答案仍只作标签，不能拼到 observation context。

下一步实现应新增有来源记录的 `ego_perception` 与 prompt 协议版本，接入
统一文本 formatter，并重建所有 ego feature cache。continuous-v1 缓存只保留
作 smoke 记录；正式 F0 验收与后续 F1 必须在修订后的相同车端合同上进行。

### 4.6 已导出的 continuous_v2_ego_gt 数据合同（历史版本）

用户已确认先使用车端 GT，之后替换 detector。代码已新增 `ego_perception.py`，
直接读取 `vehicle-side/data_info.json` 指定的原生 `label/lidar`，核对与当前
ego LiDAR timestamp 相同；不通过 cooperative PKL 导出车端对象。当前输入
包含类别、xyz、lwh、物理 heading，自车尺寸/速度/加速度/历史和统一规划提示。
原生对象速度与置信度缺失时文本为 unknown；不读取未来 track 估计速度。

新增版本 `data/Planning/continuous_v2_ego_gt/`，保持 train/val/test=929/157/480。
车端对象固定50m ROI、最近最多32个；路侧保持16个。车端来自独立 LiDAR GT
的标注覆盖范围，尚未限制前向相机 FOV，应明确报告 **Oracle ego perception**。
该阶段可以验证规划与路侧增益，不能当作真实车端检测器的性能结论。

F0=车端图像+车端GT文本+状态/提示+zero_rsu；F1 仅将 zero_rsu 替换路侧GT。
后续先固定车端GT、替换路侧 detector，单独评估路侧感知误差；再固定路侧
协议替换车端 detector，评估车端感知误差。每个来源组合分别独立训练，避免
同时替换两端后无法归因。固定权重换输入仅作分布变化诊断。

prompt 升为 `ego-perception-observation-v2`，GT/Detector 共用文本 formatter，
source 元数据不作为 VLM 的 Oracle 提示。缓存核验 prompt/version/source/frame/
timestamp，旧缓存拒绝用于新协议。19项相关测试通过，并已导出三份完整数据。

修订后的真实7B CPU smoke 已完成2条train/2条val特征与30epoch F0：训练loss
10.057→0.640，best Avg L2@1/2/3s=1.824m，仍未超过同子集恒速0.554m。
仅确认带车端GT文本的工程链路可运行；完整 F0 验收和 F1 训练尚未进行。
结果位于 `work_dirs/continuous_F0_ego_gt_smoke_20260930/`。

### 4.7 正式旧 F0 复核（2026-10-04）

自动脚本已于2026-10-02 07:43完成929 train /157 val、100 epoch的seed42 F0。
best为epoch78，主平均L2=3.9697m，4.5s FDE=9.8169m；同val恒速1.4660m，
训练均值3.8762m。CPU重载best预测与原保存预测一致，统一evaluator覆盖157/157、
missing/invalid均为0。工程完成但正式F0验收未通过，先诊断planner，不直接进入F1。
详见 [检查报告](f0_training_audit_20261004.md)，结果在
`work_dirs/continuous_F0_ego_gt_seed42/audit_20261004/`。

### 4.8 已完成的轨迹头文献核查（2026-10-04）

已筛查references中的25份PDF并核查关键训练章节，见
[研究报告](trajectory_head_training_research_20261004.md)。AURORA使用专用waypoint
token并在规划阶段联合训练语言LoRA与planner；OmniV2X冻结DINOv3视觉编码器，
先做大规模单车planner预训练再协同适配，并刻意排除当前速度等状态以防shortcut。
两者不能直接作为当前冻结聊天末token、随机MLP、929条训练数据配方的有效性证据。
建议先做state-only/模态消融和readout对照，再研究任务token+head预热+LoRA联合训练。
这些是研究建议，尚未改变M1执行协议，也未启动新训练；LoRA联合训练不能复用
detach的hidden缓存。升级后F0/F1/F2仍需同构训练，避免将模型升级误记为路侧收益。
