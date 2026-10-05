# 原生GT参考关联与因果运动补偿实验

日期：2026-10-04。前三项前置任务已实现；全量train929/val157，CPU运行。未启动VLM训练，未选择生产去重阈值。

## 实现与复现

实现入口：[run.py](../tools/association/run.py)，几何匹配：[core.py](../tools/association/core.py)。运行与字段说明：[README](../tools/association/README.md)。

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.run \
  --output-dir data/Planning/association_native_new
```

正式产物：`data/Planning/association_native_v1_final_20261004/`。`reference_pairs.jsonl`、`objects.jsonl`、`association_sweep.json`、`manifest.json`、`source_sha256.json`及`integrity.json`。输出路径已存在时拒绝覆盖。

## 参考匹配审计

全SPD共50,184条协同对象记录：10,567条双端身份唯一且一致，39,617条未知。有效双端记录中316条原始类别不同，统一类别映射后再关联。单侧ID缺失不作为负样本或真实遮挡证据。

异常包含：车端身份非唯一/找不到10条，车端token/track冲突2条，路端token/track冲突19条；同一记录可有多个问题，计数不能简单求和。

| split | 原配对可确认双端参考 | 因果消息选择后可评估参考 |
|---|---:|---:|
| train | 5867 | 5607 |
| val | 900 | 858 |

退回过去路端消息时，仅通过同sequence原生track身份迁移已验证的参考关系；当前track重复或缺失不参与。参考关系独立于匹配器，仅评估读取。

## 因果运动补偿

原生路端同track最近过去观测，0<历史间隔≤1s；统一世界坐标估计平面绝对速度，恒速外推到ego时间，再到当前ego LiDAR。无历史保留空间对齐结果，motion_valid=false。尺寸/世界朝向不外推；无接收时间，假设无额外通信延迟。采用官方SPD配对标定system_error_offset，不利用cooperative对象重新拟合偏移。

| split | 可补偿路端对象 | 无近期历史对象 | 无可用路端消息样本 |
|---|---:|---:|---:|
| train | 13993 | 1334 | 22 |
| val | 2108 | 280 | 4 |

验证集可确认参考对的平面中心差：

| 类别 | 参考数 | 仅空间对齐均值/m | 补偿后均值/m |
|---|---:|---:|---:|
| vehicle | 759 | 1.8713 | 0.9280 |
| pedestrian | 21 | 0.2693 | 0.2537 |
| cyclist | 78 | 0.7264 | 0.5288 |

这是跨视角GT一致性，不是独立运动补偿真值误差。完整报告按类别及age分组；不假设所有对象补偿后必然改善。

## 关联对照

A=旋转BEV IoU；B=BEV IoU+距离归一化+lwh对数差；C=B改为upright 3D IoU。相同类别/距离/IoU门限，Hungarian允许未匹配，最大化合法匹配数后最小化代价。距离网格vehicle1/2/4m，小目标0.5/1/2m（行人和骑行者共享小目标门限），IoU0/0.1/0.3/0.5，共108组×两种时序处理。3D IoU含高度交集，不处理完整roll/pitch多面体。

以下仅展示固定诊断参数：车辆2m，小目标1m，IoU0.1，非正式选型。

| 方法 | 时间处理 | 参考召回率 | 可判定预测精确率 | 已知错误匹配 | 未知匹配 |
|---|---|---:|---:|---:|---:|
| A | spatial | 67.60% | 100.00% | 0 | 12 |
| B | spatial | 67.60% | 100.00% | 0 | 12 |
| C | spatial | 66.90% | 100.00% | 0 | 12 |
| A | compensated | 88.46% | 99.87% | 1 | 20 |
| B | compensated | 88.46% | 99.87% | 1 | 20 |
| C | compensated | 87.53% | 99.87% | 1 | 19 |

本设置B未超过A，不能宣称组合代价更优。门限0允许无重叠框，属于诊断，不设为生产默认。精确率只覆盖有参考依据的可判定预测；未知匹配不能当作正确，也未视作错误。缺乏真实车端不可观测证据，RSU-only保留率暂不评分。

## 验证与接续

7项单元测试通过，覆盖身份缺失/冲突/重复、同sequence过去历史、匀速与静止、自车移动、时间异常、旋转与高度IoU、密集对象、未匹配和类别门限。实测产物再次核查1086个样本、17715个路端对象的因果时间/历史sequence/位移关系，9827个源文件hash一致。

前三项已完成。旧prepare/planner接口未替换，objects.jsonl是宽候选新接口，不是旧continuous缓存。后续先根据train/val诊断选定保守关联参数，再统一ROI/top-K/高可信去重，生成T1/S1共享输入；不将参考ID用于正式关联。test参数评估与VLM训练尚未执行。
