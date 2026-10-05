# UniV2X / UniMM-V2X 独立3D观测源码核查

核查日期：2026-10-04。依据为本地两库配置、detector/head/融合与转换代码，以及DrivewithVLM导出器、原生JSON和PKL。另核对官方训练评测说明。初次核查只做CPU来源统计；用户随后授权修复，已恢复完整车端PKL、生成两端无补框版本并导出原生规划v3，详见[修复验收报告](independent_observation_repair_20261004.md)。下文未实现/尚待修复描述为初次核查历史，当前状态优先看修复报告；没有训练、GPU推理或重算AP。本文支持融合计划§1.2子目标A，保持VLM直接生成轨迹与先双端GT、再逐端Detector的协议。

## 1. 单端检测架构与差异

| 项目 | UniV2X单端stage-1 | UniMM-V2X单端stage-1 |
|---|---|---|
| 配置目录 | `projects/configs_e2e_univ2x`，`univ2x_sub_{vehicle,inf}_e2e_track.py` | `projects/configs_e2e_unimmv2x`，`unimmv2x_sub_{vehicle,inf}_stg1.py` |
| 输入 | `use_camera=True, use_lidar=False` | 相同 |
| 视觉网络 | ResNet-101/DCNv2＋FPN | 同系列网络 |
| 车端主干训练 | `freeze_img_backbone=True` | `freeze_img_backbone=False` |
| BEV | 200×200，256维；TemporalSelfAttention＋SpatialCrossAttention | 同基础结构，代码支持可选MoE FFN |
| 检测/跟踪 | BEVFormerTrackHead，6层检测decoder；Hungarian、Focal＋L1 | 同系列head/loss；车端检测decoder类不同 |
| 对象query数 | 900，另有自车query | 1500，另有自车query |
| 训练队列 | 双端5帧；本项目E5-A后续训练改双端3帧 | 车4帧、路5帧 |
| 分类头 | nuScenes十类 | 相同 |
| 车端pc_range | `[-51.2,-51.2,-5,51.2,51.2,3]` | 相同 |
| 路端pc_range | `[0,-51.2,-5,102.4,51.2,3]` | 相同 |

两库均为图像→时序BEV→对象queries→3D检测/跟踪框。SPD单端转换路径只建立本端前向相机信息；multiview接口、`pts_bbox_head`、LiDAR坐标与监督不等于多个实测相机或点云输入。

**UniMM单端配置并未实际开启MoE：**BEVFormerLayer构造参数默认`is_moe=False`，仅为真时替换FFN。两个单端stage-1配置写了`num_experts=8, top_k=2`，却未传`is_moe=True`。车端使用DetrMoETransformerDecoderLayer但显式`is_moe=False`，路端使用常规decoder。不能从类名和专家数推断MoE已生效。另开MoE需要独立配置、兼容权重/训练和验证，不能擅改官方复现配置。

UniMM的协同AgentQueryFusion增加匹配后的Q/K/V注意力；匹配过滤从UniV2X按预测尺寸归一化的距离改为各轴1m限制。它是协同融合模块，不能直接当作独立D1/D2性能改进。1500queries、主干解冻和队列差异也要受控比较。当前对象前端无需接两库的motion/occupancy/planning模块。

官方UniV2X建议以推理方式缓存路端queries再训练协同模块，支持本项目冻结感知缓存的工程路线；其公开结果是协同AMOTA，不是单端stage-1检测mAP。[官方训练评测文档](https://github.com/AIR-THU/UniV2X/blob/main/docs/TRAIN_EVAL.md)。UniMM README的提升来自完整协同方法，不能归因当前单端配置。[官方仓库](https://github.com/Souig/UniMM-V2X)。

## 2. 单端PKL不等于当帧原生观测

两库`tools/spd_data_converter/spd_to_uniad.py`的单端路径先读`data_info.json['label_lidar_std_path']`，之后均执行：

1. `_generate_unvisible_annotations`：利用同track前后观测及姿态插入遮挡框，插入项写为`occluded_state=3`。
2. `_add_annotation_velocity_prev_next`：用当前和下一次观测计算GT速度，末帧填零。

补框与未来速度可作离线监督，不能直接作为当前时刻独立观测。只禁用速度不能去掉框几何对后续观测的依赖。转换器还将`valid_flag`全部置True、`num_lidar_pts`全部置1，不能当作真实点数/质量。原生标签也可能有完全遮挡对象；不能只凭`occluded_state=3`判定某框为插值。

本轮逐帧统计：加载`UniV2X/data/infos/V2X-Seq-SPD-New/<side>/spd_infos_temporal_<split>.pkl`，按frame token查本端`data_info.json`，读取`label_lidar_std_path`，比较PKL的`anno_tokens`与原生标签的`token`集合；不施加类别、ROI或置信度过滤。

| 文件 | PKL帧数 | 对应原生框数 | PKL框数 | PKL中不在当帧原生标签的token数 | 涉及帧数 |
|---|---:|---:|---:|---:|---:|
| vehicle-side/train | 空文件，0字节 | — | — | — | — |
| vehicle-side/val | 20 | 278 | 307 | 29 | 16 |
| infrastructure-side/train | 1521 | 24418 | 26056 | 1638 | 733 |
| infrastructure-side/val | 675 | 11679 | 12803 | 1124 | 421 |

后三组没有原生token被PKL丢弃。用户追问后已在内存重跑原生读取、track映射、姿态与补框函数：路端train的1638个、val的1124个额外token全部与重生成token一致，对应xyz/wlh/yaw也全部在绝对误差1e-5内一致，未出现额外token无法对应的情况。因此路端差异已确认来自该补框逻辑，而不只是根据token数量推断；尚未量化AP或规划影响。例：路端val `000912`原生19框/PKL20框；该帧track `003446`补框使用过去`000907`与未来`000928`观测，未来观测比当前晚约1.581s。

**补框是官方离线标签处理，不等于检测推理读取未来。**函数原文说明其用途为补全完全遮挡目标的轨迹；用于训练target或评测GT，不自动构成推理泄漏。只有把这些补框作为当前规划观测直接送入VLM，才不满足本项目“当帧原生独立观测”的合同。检测器在运行时以过去图像预测当前被遮挡对象仍是合法预测，应与GT补框输入区分。[官方转换源码](https://raw.githubusercontent.com/AIR-THU/UniV2X/main/tools/spd_data_converter/spd_to_uniad.py)。

**车端不完整PKL是本地产物异常，不是原生数据缺失。**按当前2196帧vehicle data_info与batch split，应为train1521帧、val675帧。实际val恰为应有validation列表的前20帧，全部在scene `0003`，最后`000965`，下一应为`000970`；该scene本身有42帧。两个文件时间均为2025-11-08 17:30:22。转换器在逐帧循环里反复覆盖保存，因此这种前缀文件与未完成生成的中间产物相符，但无日志不能确定当时为何结束。

另外`git diff`确认本地`UniV2X/tools/spd_data_converter/spd_to_uniad.py`把单端train的`info_path/mmcv.dump`语句注释掉；仓库HEAD、官方源码与UniMM版本均保留train保存。这会阻止当前脚本重建train文件，但注释本身不会创建0字节文件，不能直接认定它就是0字节的全部成因。即使train集合为空，正常序列化`{'infos': [], 'metadata': ...}`也应为非零字节。文件损坏/中断等历史原因目前未确定；本轮未恢复代码或覆盖旧PKL。

本地New两端data_info各2196帧；原始SPD车端12252帧、路端11275帧。`gen_example_data.py`包含下采样、去重及可选标注修订，cooperative另有`from_side`过滤路径；不能仅凭目录名概括标签组成。新版固定当前frame/scene split，保存标签版本和hash，不混用原始/修订标签后归因检测误差。

## 3. 当前项目需要修正的地方

`ego_perception.py:load_vehicle_gt`已从车端原生`label/lidar`读取物理rotation、lwh，速度/置信度未知；可复用其读取思路。路端`prepare.py:prepare_split`仍从基础设施PKL读取`gt_boxes/gt_names`。它虽没有用cooperative整体冒充路侧观测，却可能引入补框。新版需改为本端当帧`label/virtuallidar`；旧缓存/产物保留，不称问题已修复。旧F0是zero-RSU，不因路端来源问题直接判定其结果无效。

本地D1检测配置是单端图像推理、`other_agent_names=[]`，但训练/评测标签为cooperative PKL。预测本身仍可作为单端输出保留；原有AP不等于原生独立标签AP。车端原生train空文件、val仅20帧，需重建完整独立annotations/evaluator。D2也需分开原生当帧检测评测与含补框的tracking协议。官方权重身份正确不保证新版训练/评测协议一致，训练标签与预训练scene重叠应披露。

已有675帧预测只覆盖规划test；train/val未覆盖。补导时保持冻结split，按合法scene/timestamp推进历史，不能改变split以适配缓存。十类AP、三类诊断均值及原始预测路径仍以[既有审计](d1_d2_existing_results_audit_20261004.md)为准，本轮未改其数值。

## 4. 输出、时序与几何的复用边界

独立原生标签建立主Oracle；另设相机几何FOV Oracle对照，以区分GT→camera Detector的覆盖变化。用本端相机内外参、正深度、框投影与图像边界固定判据；pc_range/规划ROI不是相机可见范围，几何视锥不证明未遮挡。遮挡/截断/2D标签用于诊断，不作为仅Detector无法取得的输入。图像与标签点云时间分别保存。

默认先从`boxes_3d_det/scores_3d_det/labels_3d_det`导出几何候选，经validation冻结阈值、同源去重、ROI/Top-K。跟踪`boxes_3d/track_ids/track_scores`另作版本，不混用其索引/分数。300候选不是300对象。GT PKL、预测box类型和归一化回归码的尺寸/yaw/中心约定不同，须按实际类型显式转换，以非零yaw、非正方形框核对角点与投影。

UniMM `select_active_track_query`以`active_index → bbox_index → mask`对齐track embedding；`simple_test_track`仅支持bs=1顺序输入，scene变化重置。最终`unimmv2x_e2e.py`删除`bev_embed/track_query_embeddings`等大字段；现有UniV2X预测也没有这些特征。语义query实验须在删除前另导框/特征/索引，不能按检测候选行号拼接跟踪query。

两库训练路径用GT匹配和IoU筛选active queries。`torch.no_grad()`只冻结梯度，不消除GT依赖；冻结缓存必须走`return_loss=False`独立推理，不能读取matched GT index、未来轨迹或GT速度。应验证移除/改变GT不改变检测预测，GT只交独立evaluator。

训练queue长度不等于推理实际历史。按scene/timestamp推进memory，记录history tokens/reset；DDP按完整scene分片。完整过去流与规划子集稀疏流分开报告，不跨rank拆scene后声称完整时序。

缓存记录端角色、GT/Detector身份、原始/统一类别、源/ego几何、frame/scene、图像/标签/消息可用时间、pose/calibration、score/velocity及有效性、可选track/query对应索引、checkpoint/config/preprocessing/schema/split hash。source embedding区分车/路端，GT/Detector身份留审计，不作为Oracle提示。未知速度disabled且无效；不伪造协方差或点数。

只选截至当前可得的消息；无接收时间时声明无额外传输延迟假设。首版不运动补偿，坐标对齐不等于移动对象时间同步。原始宽候选与后续ROI/Top-K结果分开保存。

## 5. 实施顺序

1. 原生双端GT重导：先替换路端PKL输入，审计标签版本/时间/坐标/范围与覆盖。
2. 独立评测：补D1完整原生annotations；D2另报原生当帧指标。十类原值保留，三类正式协议另版本化，不改官方分类头来提高分数。
3. 冻结UniV2X导出：先复用已有权重与几何输出，补规划train/val合法历史；核对预训练scene重叠。
4. 固定ego GT替换RSU D2，再固定RSU协议替换ego D1；同VLM生成、同预算、各组合独立训练，AP/召回和规划L2/coverage分开。
5. UniMM受控对照：先核查环境、权重、实际MoE开关和独立AP，再决定是否替换/重训。统一标签、范围、历史、后处理，披露query数、主干冻结及训练预算差异。

两库有地图head，不代表本项目已有地图输入；Map Queries仍需地图标注、memory和decoder，首版object-only。

## 6. 关键本地证据

- [UniV2X车端配置](/home/zzn/V2X_VLM/UniV2X/projects/configs_e2e_univ2x/univ2x_sub_vehicle_e2e_track.py:106)、[路端配置](/home/zzn/V2X_VLM/UniV2X/projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py:106)。
- [UniMM车端配置](/home/zzn/V2X_VLM/UniMM-V2X-main/projects/configs_e2e_unimmv2x/unimmv2x_sub_vehicle_stg1.py:92)、[路端配置](/home/zzn/V2X_VLM/UniMM-V2X-main/projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg1.py:106)、[MoE开关](/home/zzn/V2X_VLM/UniMM-V2X-main/projects/mmdet3d_plugin/unimmv2x/modules/encoder.py:277)。
- [UniV2X单端转换](/home/zzn/V2X_VLM/UniV2X/tools/spd_data_converter/spd_to_uniad.py:487)、[UniMM补框](/home/zzn/V2X_VLM/UniMM-V2X-main/tools/spd_data_converter/spd_to_uniad.py:1003)、[未来速度](/home/zzn/V2X_VLM/UniMM-V2X-main/tools/spd_data_converter/spd_to_uniad.py:1146)。
- [UniMM推理](/home/zzn/V2X_VLM/UniMM-V2X-main/projects/mmdet3d_plugin/unimmv2x/detectors/unimmv2x_track.py:848)、[字段删除](/home/zzn/V2X_VLM/UniMM-V2X-main/projects/mmdet3d_plugin/unimmv2x/detectors/unimmv2x_e2e.py:406)。
- [原生车端读取](../tools/continuous/ego_perception.py)、[当前路端PKL读取](../tools/continuous/prepare.py:65)。
