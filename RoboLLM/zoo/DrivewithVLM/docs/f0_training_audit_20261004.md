# F0 正式训练检查（2026-10-04）

结论：工程训练正常完成，当前单 seed 的 validation 性能未通过 F0 验收。
本次仅检查及 CPU 重载评测，没有启动新训练、F1 或 test 选型。

## 实际运行与完成证据

- 启动脚本：`tools/train/wait_and_run_continuous_f0.py`。
- 监控状态：`work_dirs/continuous_F0_auto_monitor/status.json`，
  `stage=completed`，完成时间 `2026-10-02 07:43:23`。
- 正式输出：`work_dirs/continuous_F0_ego_gt_seed42/`。
- `history.jsonl` 有连续 epoch 1～100；loss 均为有限值，从3.74952降至1.16519。
- validation 最优为epoch78；epoch100主L2=4.30425m，未取最后一轮冒充best。
- `best/planner.pt`、`last/planner.pt`、配置、metrics和157条validation预测已保存。
- `run.json`：rsu_mode=zero、seed42、head_dim512、batch32、lr0.001、CPU head训练，limit=null。
- 缓存合同：929 train /157 val，prompt=`ego-perception-observation-v2`，
  ego source=`vehicle_gt`，4096维冻结VLM最后有效上下文readout。
  本次重载的FeatureDataset检查通过；这是车端独立GT验证协议。

## 同一 validation 上的结果

| 模型/基线 | L2@1s | L2@2s | L2@3s | 三时域平均L2 | 4.5s FDE |
|---|---:|---:|---:|---:|---:|
| F0 best | 1.8879 | 3.9270 | 6.0943 | 3.9697 | 9.8169 |
| 恒速度 | 0.2187 | 1.2088 | 2.9706 | 1.4660 | 6.9623 |
| 直线外推 | 0.4177 | 1.5763 | 3.4988 | 1.8309 | 7.7284 |
| 训练集均值轨迹 | 1.8192 | 3.7788 | 6.0305 | 3.8762 | 10.1085 |

单位为米。基线来自本次run.json，157条validation均有有效自车速度。
F0主指标比恒速度差约171%，也没有超过训练集均值基线。
预测有样本间变化，不能直接诊断为所有样本输出完全相同；然而泛化效果仍差。
loss下降只能说明优化在工作，不能证明规划成功。readout信息不足、优化配置和
数据分布等均需进一步诊断，本次不将任何一项断言为确定根因。

## 重载与独立评测

用 `tools.continuous.predict` 在CPU重载best，对完整157条val重新预测。
与训练保存预测的最大逐坐标差为0；随后运行统一数值evaluator：

- matched=157/157，missing=0，invalid=0，主时域有效样本157。
- 主平均L2=3.969724524m，3s FDE=6.094301904m，4.5s FDE=9.816866955m。
- 独立evaluator与训练metrics的微小差异为数值精度差异，不改变结论。
- 产物：`work_dirs/continuous_F0_ego_gt_seed42/audit_20261004/` 中的
  `reloaded_predictions.json` 与 `unified_metrics.json`。

下一步按原协议先排查F0：检查train/val误差差距、速度与history是否被readout保留、
输入/特征与轨迹可视化，再做受控优化或readout对照；当前不建议直接进入F1。
这是validation审查，未给出test、collision或真实detector成绩。

## 进一步原因诊断：VLM参与方式与信息通路

本次不是没有VLM。`cache_features.py`加载本地7B LLaVA，实际读取ego图像和
观测prompt，冻结backbone在eval/no_grad下提取4096维hidden；缓存已完整生成。
100epoch只更新ContinuousPlanner中的小规划分支；F0的AgentEncoder未使用，
有效更新参数为2,246,674，没有LoRA、VLM参数更新或轨迹loss到VLM的反传。

`add_generation_prompt=True`的模板以`<|im_start|>assistant\n`结束。
当前readout取该换行token的最终层hidden，并非专门训练的PLAN latent。
该hidden能通过因果attention读取前文，不能据token名称断言不含场景信息；
但本次没有规划监督训练VLM如何将数值状态汇总到此位置。
轨迹head只接收h_ego+zero_rsu；速度、加速度、history仅在文本中出现，
没有独立数值通路或恒速残差先验。

CPU复算best与last：

| 权重 | train主平均L2 | val主平均L2 |
|---|---:|---:|
| best(epoch78) | 2.5155 | 3.9697 |
| last(epoch100) | 1.9753 | 4.3042 |

训练改善而验证恶化，支持泛化不足/过拟合成分，不能仅归因于没有完成训练。
训练37个scene、验证7个scene；模型有效更新参数约225万。优化器为固定lr=1e-3
AdamW、weight_decay=0；这些是需要受控验证的因素，不是已确定的单一根因。

另外进行训练集拟合的闭式ridge诊断（未改正式模型，未使用test）：

- 同一冻结h上加train均值中心化、ridge，alpha=0.001/0.01/0.1/1的val主L2
  为3.3973/3.3326/3.6354/4.0589m，均未超恒速；alpha=0.001下train仅0.1199m。
  说明缓存不是完全相同或不可拟合，但跨scene可泛化数值readout仍弱；
  这些探针不能证明所有非线性decoder或其他VLM readout均无效。
- 直接读取现有合同中的vx/vy/ax/ay，train-only标准化、ridge(alpha=0.1)，
  同157条val主L2=0.6312m，4.5s FDE=3.4354m；加入mask-aware history则0.6338m。
  这是数值信息通路诊断，不是正式新baseline或test成绩；其输入继承现有
  ego-state合同，未在本次重新完成全部原始传感器因果来源审计。

证据最支持的判断是：当前冻结末token表示→小MLP的通路没有稳定利用已有
运动状态，并存在跨scene泛化问题。不能把“冻结VLM不够”当作单独确定原因。
建议先用同split做数值state旁路/恒速残差与受控优化对照，再验证readout位置、
层或pooling；最后按证据考虑LoRA/PLAN训练。图像独立贡献仍需图像消融，
本次没有重跑7B图像置换/移除实验，不能宣称模型已有效利用图像。

诊断数值保存在原输出的`audit_20261004/cause_diagnostics.json`与
`state_probes.json`；未改动正式训练权重和缓存。
