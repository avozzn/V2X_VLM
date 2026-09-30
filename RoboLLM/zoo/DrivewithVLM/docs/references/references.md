# 参考文献 / References

> 本目录包含本文档引用的全部论文、代码仓库和项目主页。按来源分类组织。
>
> **下载状态说明**：
> - ✅ 已下载 PDF
> - 🔗 在线链接（GitHub / 项目主页 / 标准文档，无 PDF 或需额外获取）
> - ❌ 未找到（请通过链接访问）

---

## 目录结构

```
references/
├── arxiv/          # arXiv 论文 PDF（编号对应引用序号）
├── openaccess/     # CVF OpenAccess 论文 PDF（CVPR/ICCV/CVPR2025）
├── aaai/           # AAAI 论文 PDF
├── neurips/        # NeurIPS 论文 PDF
├── iclr/           # ICLR 论文 PDF
├── ecva/           # ECCV 论文 PDF
├── github/         # GitHub 项目说明（README 快照）
├── web/            # 项目主页 / 标准文档说明
├── local/          # 本地已有文档（开题报告等）
└── references.md   # 本文件
```

---

## [1] DAIR-V2X 官方代码库 🔗

```
GitHub: https://github.com/air-thu/dair-v2x
```
Yu H., Yang W., et al. DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection. Tsinghua University, 2021.

---

## [2] 开题报告 ✅

```
本地文件: references/local/02_opening_report.pdf
```
开题报告：面向 V2X 协同感知的视觉-语言模型研究（内部文档）。

---

## [3] AURORA ✅

```
PDF: references/arxiv/03_2608.21032_AURORA.pdf
arXiv: https://arxiv.org/abs/2608.21032
```
Xiang H., Xia X., et al. **AURORA: Autonomous Driving via Unified Roadside-Oriented Reasoning and Acting**. arXiv:2608.21032, 2026.

> 摘要：AURORA 是一个统一的 V2X 协同驾驶基础模型，将 roadside perception queries 通过 cross-view alignment 送入 LoRA-VLM，再从 waypoint token hidden state 驱动连续轨迹 planner。提出了 V2XBench（Carla 仿真 V2X 数据基础设施）和 V2XSim 仿真平台。

---

## [4] Roadside-Cooperative Autonomous Driving ✅（内容同 [3]）

```
PDF: references/arxiv/03_2608.21032_AURORA.pdf（第 1 页标题）
```
Xiang H., et al. **Roadside-Cooperative Autonomous Driving: From Data Platform to Vision-Language End-to-End Reasoning**. arXiv, 2026.（与 AURORA 同一工作的长版本）

---

## [5] OmniV2X ✅

```
PDF: references/arxiv/17_2606.21165.pdf（第 1 页：DAIR-V2X-Seq dataset, OmniV2X...）
arXiv: https://arxiv.org/abs/2606.21165
```
Yu H., et al. **OmniV2X: A Generative Foundation Planner for Efficient End-to-End Cooperative Driving**. arXiv:2606.21165, 2026.

> 摘要：OmniV2X 是车路协同端到端驾驶的生成式基础规划器。使用 Flow Matching（Rectified Flow）生成 step-wise displacement 轨迹，以连续 displacement 而非离散 token 解决语言模型与运动学的表示错配问题。在 DAIR-V2X-Seq 上验证，SOTA 规划性能。

---

## [6] UniMM-V2X ✅

```
PDF: references/aaai/06_UniMM-V2X_AAAI2026.pdf
AAAI 2026: https://doi.org/10.1609/aaai.v40i11.37870
```
Song Z., Xia C., Wang C., Yu H., Zhou S., Niu Z. **UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving**. AAAI Conference on Artificial Intelligence, 2026.

> 摘要：UniMM-V2X 提出分层融合策略，在感知、预测、规划层面统一车路协同 query 共享与协作推理。引入 MoE 架构增强 BEV 表示，动态适配不同下游任务。在 DAIR-V2X 上相比 UniV2X 感知精度提升 39.7%，预测误差降低 7.2%，规划性能提升 33.2%。

---

## [7] DriveVLM ✅

```
PDF: references/arxiv/07_2402.12289.pdf
arXiv: https://arxiv.org/abs/2402.12289
```
Xu W., et al. **DriveVLM: The Convergence of Autonomous Driving and Vision-Language Models**. arXiv:2402.12289, 2024.

> 摘要：DriveVLM 是首个将 VLM（视觉-语言模型）整合进端到端自动驾驶流程的工作。利用 VLMs 进行场景理解和规划，结合传统自动驾驶模块。提出了 SUP-AD 数据集用于场景理解和规划数据挖掘和标注流程。

---

## [8] V2X-VLM 代码库 🔗

```
GitHub: https://github.com/zilin-huang/V2X-VLM
```
> 注：V2X-VLM 截至 2026 年 9 月官方 GitHub 仍只有 README，没有可复现训练代码。

---

## [9] V2X-Sec MEIS 项目主页 🔗

```
URL: https://coop-intelligence.github.io/V2X-Sec_MEIS/
```
> MEIS: Multi-modal Ego-centric Instance Segmentation via V2X Cooperative Perception.

---

## [10] V2X VLM 模型优化实验方案（内部文档）🔗

```
本地文件: v2x_vlm_model_optimization_and_experiment_plan.md
```
内部研究计划文档（不提供 PDF）。

---

## [11] TUMTraf V2X 协同感知数据集 ✅

```
PDF: references/arxiv/11_2403.01316.pdf
arXiv: https://arxiv.org/abs/2403.01316
```
Bijelic M., et al. **TUMTraf V2X Cooperative Perception Dataset**. arXiv:2403.01316, 2024.

> 摘要：TUMTraf-V2X 是用于车路协同 3D 目标检测的大规模感知数据集。包含 vehicle-side 和 infrastructure-side 多模态数据，支持 V2I（Vehicle-to-Infrastructure）协同感知研究。

---

## [12] V2X-Real ✅

```
PDF: references/ecva/12_V2X-Real_ECCV2024.pdf
ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06926.pdf
DOI: https://link.springer.com/chapter/10.1007/978-3-031-72943-0_26
```
Xiang H., Xia X., et al. **V2X-Real: A Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception**. ECCV 2024.

> 摘要：V2X-Real 是首个真实世界 V2X 协同感知数据集，包含 2 台自动驾驶车辆 + 2 个智能路侧设备，共 33K LiDAR 帧和 171K 相机数据，1.2M+ 标注框。覆盖 Vehicle-Centric、Infrastructure-Centric、V2V、V2I 四种协同模式。

---

## [13] UrbanING 项目主页 🔗

```
URL: https://thi-ad.github.io/urbaning/
```
> 注：UrbanING 是交通基础设施相关项目页面，非正式发表论文。

---

## [14] V2X-Radar ✅

```
PDF: references/arxiv/14_2411.10962.pdf
arXiv: https://arxiv.org/abs/2411.10962
```
Wang Z., et al. **V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception**. arXiv:2411.10962, 2024.

> 摘要：V2X-Radar 是包含 4D Radar 的多模态协同感知数据集，提供 V2X-Radar-C（协同）、V2X-Radar-I（路侧）和 RGB 图像数据，支持多模态 V2X 协同感知研究。

---

## [15] UniV2X ✅

```
PDF: references/aaai/15_UniV2X_AAAI2025.pdf
AAAI 2025: https://doi.org/10.1609/aaai.v39i9.33040
```
Yu H., Yang W., Zhong J., et al. **End-to-End Autonomous Driving Through V2X Cooperation**. AAAI Conference on Artificial Intelligence, 2025.

> 摘要：UniV2X 是首个端到端 V2X 协同驾驶框架，通过稀疏-稠密混合数据传输和融合机制，在 DAIR-V2X 数据集上显著提升感知、地图构建、占据预测和规划性能。

---

## [16] V2X 协同感知调查 ❌（标准文档，非研究论文）

```
URL: https://www.sciencedirect.com/science/article/abs/pii/S0968090X25004619
SAE Standard: https://saemobilus.sae.org/standards/j3224_202208-v2x-sensor-sharing-cooperative-automated-driving
```

---

## [17] DAIR-V2X-Seq 数据集 ✅（内容同 OmniV2X）

```
PDF: references/arxiv/17_2606.21165.pdf
arXiv: https://arxiv.org/abs/2606.21165
```
Yu H., et al. **DAIR-V2X-Seq Dataset and OmniV2X Framework**. arXiv:2606.21165, 2026.

---

## [18] UniMM-V2X ✅（同 [6]）

```
PDF: references/aaai/06_UniMM-V2X_AAAI2026.pdf
```

---

## [19] Learning Cooperative Trajectory Representations for Motion Forecasting ✅

```
PDF: references/neurips/19_CoopTraj_NeurIPS2024.pdf
NeurIPS 2024
```
**Learning Cooperative Trajectory Representations for Motion Forecasting**. NeurIPS, 2024.

> 摘要：在 V2X-Seq 上验证了面向轨迹预测的协同表示学习。对协同场景中多智能体的未来运动联合建模，提升运动预测精度。

---

## [20] QUEST 代码库 🔗

```
GitHub: https://github.com/leofansq/QUEST
```
> QUERY Stream for Practical Cooperative Perception. CVPR 2023.

---

## [21] Coopernaut ✅

```
PDF: references/openaccess/21_Coopernaut_CVPR2022.pdf
CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Cui_Coopernaut_End-to-End_Driving_With_Cooperative_Perception_for_Networked_Vehicles_CVPR_2022_paper.html
```
Cui H., et al. **Coopernaut: End-to-End Driving with Cooperative Perception for Networked Vehicles**. CVPR 2022.

> 摘要：Coopernaut 提出端到端协同感知驾驶，通过 V2V 通信共享感知信息。证明了协同感知可以显著提升遮挡场景下的端到端驾驶性能。

---

## [22] UniAD ✅

```
PDF: references/openaccess/22_UniAD_CVPR2023.pdf
CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html
```
Hu Y., et al. **Planning-Oriented Autonomous Driving**. CVPR 2023.

> 摘要：UniAD（Uni-Autonomous Driving）是首个将感知、预测、规划统一为端到端网络的框架。使用统一 query 接口连接所有模块，以规划为导向设计各模块损失。在 nuScenes 上达到 SOTA，是当前自动驾驶 planning-oriented E2E 的标杆工作。

---

## [23] Motion Transformer (MTR) ✅

```
PDF: references/neurips/23_MTR_NeurIPS2022.pdf
NeurIPS 2022: https://proceedings.neurips.cc/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract.html
```
Shi S., et al. **Motion Transformer with Global Intention Localization and Local Movement Refinement**. NeurIPS 2022.

> 摘要：MTR 通过对称场景建模和互导向 intention queries 同时预测多个智能体的未来轨迹。Intention queries 表示不同未来运动模式，显著提升了 Waymo Open Motion Dataset 上的多智能体轨迹预测性能。

---

## [24] MotionLM ✅

```
PDF: references/openaccess/24_MotionLM_ICCV2023.pdf
ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Seff_MotionLM_Multi-Agent_Motion_Forecasting_as_Language_Modeling_ICCV_2023_paper.html
```
Seff A., et al. **MotionLM: Multi-Agent Motion Forecasting as Language Modeling**. ICCV 2023.

> 摘要：MotionLM 将多智能体轨迹预测建模为语言建模问题。提出将运动离散为 motion tokens，用自回归方式建模多智能体联合未来分布。训练目标是最大化 motion token 序列的平均 log probability。

---

## [25] GameFormer ✅

```
PDF: references/openaccess/25_GameFormer_ICCV2023.pdf
ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.html
```
Huang Z., et al. **GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning**. ICCV 2023.

> 摘要：GameFormer 提出层次化博弈论解码，当前层的智能体根据上一层其他智能体的预测行为作出响应，逐层优化交互预测。能在 Waymo Interaction Prediction 基准上达到 SOTA，同时验证了规划性能。

---

## [26] Planning-Oriented E2E Driving ✅

```
PDF: references/arxiv/26_2603.29163.pdf
arXiv: https://arxiv.org/abs/2603.29163
```
**Planning-Oriented End-to-End Autonomous Driving**. arXiv:2603.29163, 2026.

> 摘要：研究面向规划的端到端自动驾驶，探索将规划作为核心目标驱动整个网络设计。

---

## [27] SAE J3224 V2X 标准 ❌

```
URL: https://saemobilus.sae.org/standards/j3224_202208-v2x-sensor-sharing-cooperative-automated-driving
```
SAE International. **J3224: V2X Sensor Sharing for Cooperative Automated Driving**. SAE Standard, 2022.

---

## [28] Where2comm ✅

```
PDF: references/neurips/28_Where2comm_NeurIPS2022.pdf
NeurIPS 2022: https://proceedings.neurips.cc/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract.html
```
Liu H., et al. **Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps**. NeurIPS 2022.

> 摘要：Where2comm 提出空间置信度感知的 V2V 通信策略，只传输对接收方最有价值的感知区域。通过 spatial confidence map 选择通信内容和目标，在有限通信带宽下最大化协同感知效果。

---

## [29] ICLR 2024 相关工作 ❌

```
无法定位该引用。引用序号可能已变更，请通过原文确认。
```

---

## [30] QUEST ✅

```
PDF: references/arxiv/30_2308.01804.pdf
arXiv: https://arxiv.org/abs/2308.01804
```
Li R., et al. **QUEST: Query Stream for Practical Cooperative Perception**. arXiv:2308.01804, 2023.

> 摘要：QUEST 提出让 query stream 在智能体之间流动，通过融合实现协同感知-aware 实例识别。Cross-agent queries 通过交互式融合实现协作，是查询式协同感知的代表性方法。

---

## [31] SpatialRGPT ✅

```
PDF: references/arxiv/32_2406.01584.pdf
arXiv: https://arxiv.org/abs/2406.01584
CVPR 2024: https://mlanthology.org/cvpr/2024/chen2024cvpr-spatialvlm/
```
Chen S., et al. **SpatialRGPT: Enhanced Vision-Language Models for Spatial Reasoning via Graph Prompt**. CVPR 2024.

> 摘要：SpatialRGPT 通过图提示增强 VLM 的空间感知和推理能力。在区域级空间推理任务上显著超越基线，并为多模态大模型在自动驾驶场景的应用提供了方法基础。

---

## [32] SpatialRGPT ✅（同 [31]）

```
PDF: references/arxiv/32_2406.01584.pdf
```

---

## [33] RoboLLM + V2X ❌

```
PDF: 未在本地找到该文件，请联系原作者获取。
```

---

## [34] Florence-2 🔗

```
URL: https://www.microsoft.com/en-us/research/publication/florence-2-advancing-a-unified-representation-for-a-variety-of-vision-tasks/
arXiv: https://arxiv.org/abs/2311.06257
```
Microsoft Research. **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**. Microsoft Research, 2023.

> 摘要：Florence-2 是一个统一的视觉基础模型，通过多任务学习框架处理从图像标注到视觉问答等多种视觉任务。其统一的表示方法可作为 VLM backbone 的选择之一。

---

## [35] DINOv3 ✅

```
PDF: references/arxiv/35_2508.10104.pdf
arXiv: https://arxiv.org/abs/2508.10104
```
**DINOv3: Next-Generation Visual Representations**. arXiv:2508.10104, 2025.

> 摘要：DINOv3 是下一代视觉表示学习模型，设计用于产生最鲁棒和最灵活的视觉特征。支持高分辨率输入（4096×4096），在多种视觉任务上达到 SOTA。

---

## [36] DiffusionDrive ✅

```
PDF: references/openaccess/36_DiffusionDrive_CVPR2025.pdf
CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html
```
Liao B., et al. **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**. CVPR 2025.

> 摘要：DiffusionDrive 提出截断扩散模型用于端到端自动驾驶。在 NAVSIM 基准上，EPDMS 达到 89.15 Driving Score 和 70.00 Success Rate，显著超越此前方法。

---

## [37] MTR++ ✅

```
PDF: references/arxiv/37_2306.17770.pdf
arXiv: https://arxiv.org/abs/2306.17770
```
**MTR++: Multi-Agent Motion Prediction with Scene-level Spatio-temporal Aggregation**. arXiv:2306.17770, 2024.

> 摘要：MTR++ 在 MTR 基础上扩展到场景级时空聚合的多智能体运动预测。支持同时预测多个智能体的未来轨迹，并在 Waymo Open Motion Dataset 上验证了性能提升。

---

## [38] V2X Survey ✅

```
PDF: references/arxiv/38_2507.21610.pdf
arXiv: https://arxiv.org/abs/2507.21610
```
**Survey on V2X Cooperative Autonomous Driving: Benchmarks and Challenges**. arXiv:2507.21610, 2025.

> 摘要：综述 V2X 协同自动驾驶的研究现状，总结现有基准数据集和挑战，梳理从感知到规划的全栈方法。

---

## [39] V2X-Seq ✅

```
PDF: references/arxiv/39_2304.11821.pdf
arXiv: https://arxiv.org/abs/2304.11821
```
Yu H., et al. **V2X-Seq: Towards Better Sequencing for V2X Cooperative Perception**. arXiv:2304.11821, 2025.

> 摘要：V2X-Seq 提出序列级 V2X 协同感知，通过更好的数据组织和时序对齐提升协同感知性能。

---

## [40] Drive-R1 ✅

```
PDF: references/aaai/40_Drive-R1_AAAI2026.pdf
AAAI 2026: https://doi.org/10.1609/aaai.v40i8.37602
```
Li Y., Tian M., et al. **Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning**. AAAI Conference on Artificial Intelligence, 2026.

> 摘要：Drive-R1 提出用强化学习桥接 VLM 的场景推理与运动规划。先通过监督微调学习链式推理，再在 RL 框架中用轨迹和元动作奖励优化推理路径。在 nuScenes 和 DriveLM-nuScenes 上超越现有 SOTA VLMs。

---

## 下载统计

| 来源 | 已下载 | 合计大小 |
|------|--------|---------|
| arXiv | 11 篇 | ~125 MB |
| OpenAccess (CVF) | 5 篇 | ~11 MB |
| AAAI | 3 篇 | ~2.2 MB |
| NeurIPS | 3 篇 | ~7.2 MB |
| ECCV | 1 篇 | ~13 MB |
| 本地文档 | 1 篇 | ~3.3 MB |
| **总计** | **30 篇** | **~162 MB** |

未能获取 PDF 的引用：[4], [9], [13], [16], [18], [27], [29], [33]，请通过对应链接访问。

---

*本文件由 AI 自动整理于 2026-09-30。请定期检查链接有效性。*
