# 原生双端 GT 关联与因果运动补偿

从项目根目录运行，CPU，不加载 VLM、不启动训练：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.run \
  --output-dir data/Planning/association_native_new
```

依赖 numpy、scipy、shapely（现有 robollm 环境具备）。输出目录必须不存在，防止覆盖。
默认读取原生 SPD 与既有规划 split，运行 train 929 / val 157；`--limit 3`
用于 smoke，`--splits test` 仅供后续参数锁定后评估，不用于选参数。

## 输出与接口

- `reference_pairs.jsonl`：全 SPD cooperative 每个对象的两端身份核查结果；只有双端 frame/sequence/timestamp/token/track 一致且唯一的记录有效。
- `objects.jsonl`：逐规划样本原生对象；车端 `box`、路端 `raw_box/spatial_box/box` 均为 `[x,y,z,l,w,h,yaw]`。后两种位于当前 ego LiDAR；`raw_box` 位于源路端。包含历史来源、world/ego velocity、motion_valid、age；没有未来轨迹或 cooperative 对应 ID。
- `association_sweep.json`：A/B/C、空间/补偿、距离/IoU参数扫描；整体、类别及消息年龄分组。
- `source_sha256.json`：实际读取的原生标签、配对元数据及标定文件 hash。
- `manifest.json`：输入规划文件和代码 hash、来源审计、运动统计及跨视角中心差异分组。

该格式是宽候选离线实验接口，**不能直接作为旧 continuous planner 缓存**；未做 ROI/top-K，也尚未选择正式去重参数。T1/S1 后续共享一份从此接口产生的选择/去重结果。

## 因果与关联约定

使用不晚于 ego 的配对/之前路端帧；速度来自同 sequence/track 最近一次更早原生观测，历史间隔最多 1 秒；先到世界坐标估计平面绝对速度，外推后到当前 ego。无历史则仅空间转换，速度无效。恒速补偿不估计 yaw rate。

采用官方 SPD 标定约定，在变换后的 ego 中加 data_info 的 system_error_offset；不从 cooperative 对象反求标定。没有实际接收时间，假设无额外传输延迟。

所有方法统一类别和距离门限。A 最小化 `1-BEV IoU`；B 增加归一化中心距离和 lwh 对数差；C 用 yaw-aware upright 3D IoU 替换 B 的 BEV IoU。3D IoU 是旋转 BEV 交面积乘高度交集，不计算带 roll/pitch 的任意多面体交集。门限禁止边，dummy 匹配允许不匹配；先最大化合法匹配数再最小化代价。IoU 门限 0 允许无重叠候选，作为诊断而非生产默认。

参考对应仅用于评估；旧配对退回历史消息时，参考通过同 sequence 内的原生 track 身份迁移，缺失/重复身份不计参考。任何预测若涉及已知对应端点，可判定正误；其余为未知。`precision_on_judgable_predictions` 不是全部预测精确率，必须同时报告 `unknown_predictions`。

无车端 ID 不证明车端不可观测，RSU-only 保留率暂未评分。中心差是两端 GT 一致性指标，不能称独立运动补偿真值误差。

测试：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m unittest discover \
  -s tests -p test_native_association.py
```


## 尺寸稳定、软关联与尺寸融合（2026-10-04已实现）

`stable_fusion.py`提供双端本端因果历史中位数、补偿框、软Hungarian、保守验收及select/blend尺寸政策；`evaluate_stable.py`运行A0–A4，并把纯算法产物与cor评估隔离。当前验证后冻结A1；A2–A4保留作对照，不代表验收通过。

使用不存在的新目录，从项目根目录依次运行：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.evaluate_stable \
  --output-dir data/Planning/stable_fusion_new_trainval
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.freeze_stable \
  --trainval-dir data/Planning/stable_fusion_new_trainval \
  --output data/Planning/stable_fusion_new_trainval/frozen_selection.json
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.evaluate_stable \
  --splits test --output-dir data/Planning/stable_fusion_new_test \
  --frozen-selection data/Planning/stable_fusion_new_trainval/frozen_selection.json
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.report_stable \
  --trainval-dir data/Planning/stable_fusion_new_trainval \
  --test-dir data/Planning/stable_fusion_new_test \
  --output-dir evaluation_results/stable_fusion_new_report
```

`--limit 3 --no-evaluation`用于不读cor标签的探针；配对元数据/标定仍必需。缺失cor不影响几何算法本身，但不能产生参考指标。test要求先有冻结记录，不能用于选政策。

- `objects.jsonl`：原生对象、原始/稳定尺寸、因果历史与补偿；无cor对象字段。
- `decisions.jsonl`：五组配对、拒绝原因、融合输出、来源位；不含评估真值。
- `evaluation.jsonl`：单独cor参考评估；`summary.json`含分组、固定共同配对队列与场景bootstrap。
- `reference_pairs.jsonl`、`source_sha256.json`、`manifest.json`：身份审计与可追溯来源。
- `frozen_selection.json`：val选型结果与算法/依赖hash。

历史与尺寸统计使用GT track，是Oracle实验；Detector必须换本端预测轨迹。尺寸MAD不是标定协方差，cor尺寸不是独立实物测量。宽候选未做ROI预算，也未接入T1/S1生成，不能直接替代旧planner缓存。

本次真实产物与图表：`evaluation_results/stable_fusion_v1_20261004/report.md`；完整检查记录为同目录`integrity.json`。28项相关测试通过，包括8项新几何算法测试：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m pytest \
  tests/test_stable_fusion.py tests/test_native_association.py \
  tests/test_native_observation_repair.py tests/test_rsu_continuous_pipeline.py -q
```


## 放宽关联V2（train/val，已实现）

独立新模块`relaxed_fusion.py`保留旧算法不动。A1/L1/L2统一原始尺寸；L1取消验收距离减半，L2对≤100ms的未知运动消息要求cost≤0.45、BEV IoU≥0.1才放行。朝向和匹配歧义要求仍保留。非可信对象保持双端独立，无尺寸融合。

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.association.evaluate_relaxed \
  --source-dir data/Planning/stable_fusion_v1_trainval_20261004 \
  --output-dir data/Planning/relaxed_fusion_new_trainval
```

只接受已有审计train/val缓存；执行前重新核验所有原生源文件hash。A1逐帧对照旧配对，纯算法decision与cor评价分开；每次用新的输出目录。完整真实结果：`data/Planning/relaxed_fusion_v2_trainval_20261004/report.md`。当前仍保留A1，不自动改变T1/S1或旧planner。6项新增测试覆盖完整距离门限、短延迟未知速度、几何证据不足、消息年龄、歧义、尺寸及空输入。
