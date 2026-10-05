# T1/S1共同对象缓存与自回归接口实施记录

日期2026-10-04。用户后续决定：GPU实验由用户自行运行，提供脚本；不自动训练或等待空闲卡。

## 完成与未完成

已实现`tools/object_vlm/`共同导出、T1文本、S1共享encoder/位置/来源/质量编码、原始车端图像tokens保留、答案CE、生成/KV cache、LoRA训练、validation、checkpoint与用户启动脚本。31项相关CPU测试通过，其中新接口10项。真实processor/图像训练标签边界与tiny Qwen/SigLIP梯度/缓存/LoRA重载均测试。

真实7B首次smoke在加载时显存不足：0–3卡被其他任务占用，4–7也在使用。失败输出保留在`work_dirs/T1_object_smoke_v1_20261004`和`S1_object_smoke_v1_20261004`，不算成功smoke。当前没有T1/S1训练成绩，正式训练和真实smoke由用户运行；不启动等待队列。

## 数据与公平对照

`data/Planning/dual_object_a1_v1_20261004/`：929/157/480帧，来自双端原生GT和现有规划split。A1因果补偿/统一坐标/硬IoU关联，配对保留ego原始尺寸与中心，50m ROI、最多32ego贡献＋16额外RSU对象。两组同一对象、顺序、字段及速度禁用策略。

| split | 对象总数 | 双端配对对象 | 额外路端对象 | 空对象帧 | 预算截断对象 |
|---|---:|---:|---:|---:|---:|
| train | 10459 | 3050 | 4662 | 21 | 343 |
| val | 1492 | 402 | 559 | 7 | 7 |
| test | 5954 | 1491 | 2599 | 11 | 116 |

补偿在范围选择之前；独立audit记录原始身份与历史，输入不含这些字段。导出不读cooperative对象labels。逐帧验证split hash、来源坐标、消息和历史截止时间、有限数值、尺寸/类别/来源/预算与未来监督隔离。test只导出，未评测或用于选型。

T1/S1均保留同一车端图像与自车state/history文本；S1删除对象文本，以数值geometry/source/quality生成每对象一个soft token。未来九点为0.5至4.5s答案监督，所有观测labels=-100。图像实际729 tokens，train/val审计T1最长6521/S1最长1604，统一7168上限；禁止截断。

## 用户启动

从项目根目录：

```bash
T1_GPU=0 S1_GPU=1 bash tools/object_vlm/run_t1_s1.sh
```

将0/1改成用户安排的两张物理卡。脚本并行两组真实smoke；都通过之后才从基础LLaVA重新初始化，运行各6epoch。有效batch4、seed42、LoRA8/8、lr2e-4，视觉模块冻结，S1 encoder与LoRA联合训练。GPU资源不足会退出，不终止其他任务。具体仅smoke/续接训练/预测命令、输入合同和失败处理见[运行说明](../tools/object_vlm/README.md)。

每epoch统一val报告主平均L2、逐时域L2、4.5sFDE、解析率、时延、输入长度和显存；先比较coverage再选L2。不要用低解析率的有效子集宣称规划提升。单seed为探索结果，未回答RSU相对ego-only增益。

旧I1连续接口未改成generate，旧MMdrive运行不能当作T1，旧缓存和checkpoint未覆盖。BEV、Map、StateEncoder、动态范围、尺寸融合和连续head仍留后续。

2026-10-04用户四卡修订：正式训练改为TRAIN_GPUS=0,1,2,3，先T1四卡6epoch，再S1四卡6epoch，均从基础初始化，有效batch仍4；smoke仍单卡并行且两组都成功才训练。torchrun梯度同步、最后singleton不重复/丢样本，validation四卡分片/汇总、rank0保存；不自动启动GPU，用户执行脚本。此前两组正式单卡并行描述已被本修订取代。

2026-10-04最新实测：work_dirs/T1_S1_objects_4gpu_smoke_20261004中T1/S1真实7B单卡梯度/对象敏感性/重载生成检查均通过，双组0–3四卡9train/8val、3更新、分片验证与保存均通过；34项相关CPU测试通过。正式6epoch尚未启动。短probe严格轨迹解析0/8，不能作为规划效果。正式脚本为ACTION=train TRAIN_GPUS=0,1,2,3 RUN_ROOT=work_dirs/T1_S1_objects_4gpu_smoke_20261004 bash tools/object_vlm/run_t1_s1.sh；T1先四卡6epoch，再S1四卡6epoch，仍由用户启动。结果见该目录report.md和smoke_summary.json。

2026-10-05正式结果确认：T1/S1均完成四卡6epoch、1398次更新，六个epoch checkpoint与optimizer文件齐全，宿主机无残留训练进程。T1最佳epoch5，val主平均L2=0.6925203365m；S1最佳epoch6=0.6358840247m；两者最佳均157/157解析成功，4.5sFDE分别3.7897720543/3.5293594817m。单seed42验证集探索，不证明RSU增益；test未评测。产物work_dirs/T1_S1_objects_4gpu_smoke_20261004/T1_train和S1_train下complete.json/best_checkpoint.json/history.jsonl。


2026-10-05 test评测脚本已提供（尚未运行GPU）：`tools/object_vlm/eval_t1_s1.sh`和`evaluate.py`按val冻结T1 epoch5/S1 epoch6，默认0–3四卡依次生成两组完整480帧test；复用训练时期的生成/解析/指标实现，核对checkpoint、cache、base和代码hash，输出逐帧预测、指标及对比report.md。启动命令：`EVAL_GPUS=0,1,2,3 bash tools/object_vlm/eval_t1_s1.sh`。用户自行启动，不自动评测。最终CPU预检查通过，证据`evaluation_results/T1_S1_test_preflight_final_20261005/frozen_selection.json`；新增3项测试通过，覆盖coverage优先选型、拒绝未完成训练、全部解析失败时不产生虚假误差收益。没有新test成绩。


2026-10-05 test长度超限修复：用户首轮test评测在T1单帧prefill 7180>7168时退出，尚无完整test成绩。原预检查漏审计test提示长度；7168为训练包装器guard，基础Qwen配置支持32768。evaluate.py现对完整480帧两种表示的观测提示进行CPU长度审计（不读轨迹答案），记录T1最大7180/1帧超guard、S1最大1242/0帧超guard；统一评测guard提升到8192，额外检查512生成预算在模型context内。checkpoint加载后显式覆盖model.max_length，原model.py及训练权重/代码保持不变；不截断、不删对象、不重新选epoch。修复NCCL销毁后异常hook调用get_rank的连带报错。4项评测CPU测试通过，最终预检查见evaluation_results/T1_S1_test_context_preflight_20261005/frozen_selection.json。用户仍用原命令启动，自动新输出目录，旧失败目录保留；本次未启动GPU。


2026-10-05 T1/S1完整test完成：evaluation_results/T1_S1_test_20261005_114004，两个complete.json均480帧、同一冻结selection hash，均480/480合法解析。T1 val-selected epoch5：L2@1/2/3s=0.1098818/0.5764141/1.5094839m，主平均0.7319266m，FDE4.5=3.824675m；S1 val-selected epoch6：0.1029771/0.5642284/1.4801853m，主平均0.7157969m，FDE4.5=3.794841m。S1主平均下降2.2037%（绝对0.01613m），小于val的8.18%，不能表述显著或稳定优势；单seed且无ego-only对照，不能推断RSU收益。tokens3076.45→1198.425（少61.05%）；生成阶段计时1692.68s→1550.50s（少8.40%，不含加载）；峰值allocated显存21,454,255,616→17,272,263,168bytes（少19.49%）。重新由逐帧预测和原始GT计算三时域平均与保存指标一致；S1在254/480帧、12/19场景改善，其余226帧/7场景退化，paired_analysis.json已保存。本轮仅分析完成产物，无新增训练/评测。建议保留S1作为紧凑接口候选，多seed确认精度差异；若研究路侧价值需补同构ego-only，不用test调参。


2026-10-05长时域误差诊断：依据同480帧预测重算，horizon_error_diagnosis.json保存全九步mean/median/p90及XY分量。S1 L2@2/3/4.5s=0.5642/1.4802/3.7948m，median=0.4841/1.2357/3.2063m；467/480帧3s劣于2s，不只是少数离群。平均|dx|=0.4285/1.0460/2.5213，平均|dy|=0.2502/0.7495/2.1451，非只有纵向。回顾性按GT末段(3→4.5s)方向相对当前ego前向角绝对值>0.15rad粗分（不等同正式转弯类别/意图标签）：239帧该组S1 3s/4.5s=1.8485/4.8415，T1=1.8167/4.7549；241帧其余组S1=1.1150/2.7569，T1=1.2049/2.9021。S1改善主要在该近直行粗组，曲线/转向难题未解决。固定已有state的CV诊断2/3/4.5s=1.1899/2.9029/6.8041，CA=.8273/1.9488/4.8657；T1/S1均优于这些简单外推，不能称仅复制恒速。误差随时域非线性增长符合运动变化难度，但没有定位到具体模块的因果证据。当前监督为答案token CE，无连续坐标/曲率辅助loss；StateEncoder未接入，两组同状态文本。后续优先审计速度/加速度/历史数据实际来源、转向场景与连续几何监督消融；未新增训练/推理。
