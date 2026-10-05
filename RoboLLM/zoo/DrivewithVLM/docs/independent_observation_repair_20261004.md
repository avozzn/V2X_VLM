# 独立双端观测数据转换修复与验收

日期：2026-10-04。已完成CPU数据检查、转换修复、完整PKL重建、原生双端规划数据导出和13项相关测试。未运行Detector/VLM推理、AP重评或训练。

## 1. 下载与转换的结论

参照本地[DAIR-V2X SPD目录说明](/home/zzn/V2X_VLM/DAIR-V2X/docs/get_started_spd.md)与[SPD单端loader](/home/zzn/V2X_VLM/DAIR-V2X/v2x/dataset/dair_v2x_for_detection.py:305)，单端GT由本端data_info中的`label_lidar_std_path`或`label_camera_std_path`读取。LiDAR标签以点云时间为准，camera标签以图像时间为准；本次Oracle继续使用独立LiDAR GT，不改变传感器setting。

本轮检查原始SPD和New子集的每个data_info `*_path`引用是否存在、非空；所有引用JSON均解析。

| 数据目录/端 | 帧数 | 引用路径检查次数 | JSON解析次数 | 缺失/空文件/JSON错误 |
|---|---:|---:|---:|---:|
| 原始SPD vehicle | 12252 | 98016 | 73512 | 0/0/0 |
| 原始SPD infrastructure | 11275 | 78925 | 56375 | 0/0/0 |
| New vehicle | 2196 | 17568 | 13176 | 0/0/0 |
| New infrastructure | 2196 | 15372 | 10980 | 0/0/0 |

没有发现数据引用漏下载的证据；本次故障在派生文件生成/保存，不需要重新下载。此检查覆盖索引引用及JSON内容，不是对全部图片/点云做归档checksum认证。

车端旧PKL是0字节train与val前20帧；转换器逐帧覆盖写入会留下中间前缀，本地train保存又被注释。确切历史中断原因无日志支持，不猜测由谁/哪次操作造成。

## 2. 已修复代码

- [UniV2X转换器](/home/zzn/V2X_VLM/UniV2X/tools/spd_data_converter/spd_to_uniad.py)：恢复单端train保存；将train/val保存移到完整转换循环之后，临时文件写完后`os.replace`逐文件原子发布。默认保留官方补框监督，新增单端`--no-interpolate`供原生当帧框版本。
- [规划prepare](../tools/continuous/prepare.py)：路端PKL仅用于frame/time/pose与前驱链，框和类别直接来自选定帧原生`label/virtuallidar`，严格核对scene与timestamp；不读取PKL `gt_boxes/gt_names/gt_velocity`作为观测。保留原有解析变换、因果配对、ROI=50m、ego32/RSU16与disabled velocity。
- [车端读取](../tools/continuous/ego_perception.py)：继续读原生`label/lidar`，增加标签hash、图像timestamp、annotation token、track ID及遮挡/截断审计信息；formatter不把这些GT身份提示写进模型文本。

`--no-interpolate`控制框来源，PKL的`gt_velocity`仍是由下一观测产生的离线监督target，metadata显式注明`next_observation_supervision_only`。不能直接作为Oracle速度输入；新版规划不读取该字段。单端与cooperative标签协议分开，不修改原生JSON或旧cooperative标签。

## 3. 已生成和发布的PKL

两个新目录均为每端train1521帧、val675帧；帧token唯一，按原生标签逐帧验收：

| 端/split | 官方补框兼容版本框数 | 原生当帧版本框数 | 原生版本额外框 |
|---|---:|---:|---:|
| vehicle/train | 18985 | 17336 | 0 |
| vehicle/val | 9590 | 8770 | 0 |
| infrastructure/train | 26056 | 24418 | 0 |
| infrastructure/val | 12803 | 11679 | 0 |

兼容版本：`/home/zzn/V2X_VLM/UniV2X/data/infos/V2X-Seq-SPD-Repaired-20261004/`。
原生当帧版本：`/home/zzn/V2X_VLM/UniV2X/data/infos/V2X-Seq-SPD-Native-20261004/`。
原生版本`anno_tokens`逐帧与原生JSON token集合完全相等。

验收后已将兼容版本车端train/val发布到原路径`data/infos/V2X-Seq-SPD-New/vehicle-side/`，使旧路径不再指向损坏文件。旧0字节/20帧文件完整备份在该目录的`backup_before_repair_20261004/`。路端原路径与cooperative文件未覆盖。原路径兼容版仍含官方补框，不把它冒充原生观测。

已有D1 E5-A配置仍保留历史cooperative训练/评测来源，既有分数不改写。独立annotations已可用，正式原生AP需另对齐nuScenes JSON评测GT并重评，不能只改ann_file后声称原生指标已完成。

## 4. 新版规划数据与验证

新数据：`data/Planning/continuous_v3_native_gt/{train,val,test}.json`及`manifest.json`；prepare默认输出已改到该新版本，旧v2保留。

| split | 样本数 | RSU有效对象总数（50m/K16后） | 相对旧v2对象几何变化样本数 |
|---|---:|---:|---:|
| train | 929 | 7000 | 140 |
| val | 157 | 954 | 22 |
| test | 480 | 3692 | 115 |

逐样本验证新旧token顺序、规划targets、RSU source_token/timestamp相同；新RSU观测时间均不晚于ego。消息年龄仍可能约2.186s，未运动补偿。对象变化来自原生来源重导；不将其当作规划收益或AP变化。

13项测试通过：新增原生路端来源/timestamp校验、PKL框NaN污染不影响原生输出、完整train/空val可序列化、模拟写盘失败时旧文件保留；既有车端读取、框坐标/yaw与输入隔离测试继续通过。测试命令：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m pytest \
  tests/test_native_observation_repair.py tests/test_rsu_continuous_pipeline.py -q
```

必要复现入口：

```bash
# 规划数据：使用新的输出目录，拒绝覆盖已存在split文件。
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.continuous.prepare \
  --output-dir data/Planning/continuous_v3_native_gt_another_version

# 独立原生PKL：从UniV2X目录执行，save-root选新的版本目录。
/home/zzn/anaconda3/envs/univ2x/bin/python tools/spd_data_converter/spd_to_uniad.py \
  --data-root datasets/V2X-Seq-SPD-New --save-root data/infos/SPD-native-another-version \
  --v2x-side vehicle-side --no-interpolate
```

路端转换将`--v2x-side`改为`infrastructure-side`。完整转换脚本拒绝已存在版本目录；见下述rebuild_infos.py。本次修复只恢复转换与原生观测来源，不代表双端soft tokens或语言生成训练已接入。

## 5. 证据产物

路径均相对项目根：`evaluation_results/independent_observation_repair_20261004/`。

- `data_integrity.json`：四组引用检查与错误列表。
- `rebuild_infos.py`、`rebuild.log`、`rebuild_summary.json`：完整两端兼容/原生版本重建和逐帧token验收。
- `published_vehicle_files.json`：旧文件备份路径、修复前后hash与发布帧数。
- `planning_verification.json`：新旧split/targets/配对一致性与对象变化统计。
- 新数据`data/Planning/continuous_v3_native_gt/manifest.json`：来源、schema、数量、hash及age范围。


## 6. 关联实验兼容性复核（2026-10-04）

本轮独立复核：双端Native目录train各1521帧、val各675帧，所有anno_tokens与当帧原生JSON一致，并逐对象比较xyz/wlh/编码yaw，均通过；对象数分别为vehicle17336/8770、infrastructure24418/11679。发布到原路径的车端PKL实际可读取，帧token唯一且帧数1521/675。单端保存函数位于完整转换循环之后，临时文件与目标同目录，os.replace逐文件原子发布（不是train/val两文件整体事务）。

新版规划train/val/test=929/157/480，逐样本tokens/scene/timestamp/targets/RSU配对与旧版一致；原生标签hash/token核查、非未来消息、有限几何与禁用速度通过。几何变化样本140/22/115，RSU有效对象7000/954/3692，与修复报告一致。

此前tools/association/run.py实验不读取vehicle-side派生PKL，直接读取原生双端JSON和标定；旧continuous_v2仅用于token/split/ego时间戳，不读取其中对象框。association_native_v1_final_20261004记录的9827个原生源文件hash均未变化，train/val规划列表hash也未变，因此车端损坏train及本次派生PKL修复不影响此前关联/运动补偿结果，不需要为此重跑该对照。

复跑tests/test_native_observation_repair.py、tests/test_rsu_continuous_pipeline.py、tests/test_native_association.py，共20项通过（原修复13项+关联7项）。未重评AP或训练。continuous_v3_native_gt仍是无运动补偿版本；因果补偿位于独立association输出，未自动接入prepare或T1/S1。
