# T1/S1 自回归对象输入

2026-10-04：接口及四卡训练已实现，34项相关CPU测试通过。用户授权后，真实7B双组单卡smoke和0–3四卡短训练均通过，尚未启动正式6epoch训练。短probe严格解析率0/8，不代表规划效果；详见`work_dirs/T1_S1_objects_4gpu_smoke_20261004/report.md`。

## 运行

从项目根目录，指定两张实际可用的物理GPU：

```bash
TRAIN_GPUS=0,1,2,3 bash tools/object_vlm/run_t1_s1.sh
```

默认`ACTION=all`，两组单卡并行smoke（T1_GPU默认0、S1_GPU默认1），然后依次执行T1/S1四卡短训练检查（每组9train/8val、1epoch、3次更新）。全部通过之后，正式训练使用TRAIN_GPUS默认0,1,2,3：先T1四卡6epoch，结束再S1四卡6epoch。每个正式实验从相同基础LLaVA初始化，不复用smoke更新。四卡各处理一个样本，梯度同步，有效batch仍为4、不额外累积；输出目录自动含运行时间。GPU剩余显存低于35GiB时直接报错退出，不停止任何其他进程。

先只验证接口：

```bash
ACTION=smoke T1_GPU=0 S1_GPU=1 RUN_ROOT=work_dirs/T1_S1_objects_manual_smoke bash tools/object_vlm/run_t1_s1.sh
```

成功后继续正式训练（复用同一RUN_ROOT，不复用smoke权重）：

```bash
ACTION=train TRAIN_GPUS=0,1,2,3 RUN_ROOT=work_dirs/T1_S1_objects_manual_smoke bash tools/object_vlm/run_t1_s1.sh
```

`ACTION=smoke`包含单卡接口检查和四卡短训练检查，不运行正式6epoch。脚本拒绝smoke同卡并行；TRAIN_GPUS要求四个不同编号。正式T1/S1顺序使用同一组四卡。新smoke/all运行目录必须不存在；train要求已成功的两组smoke，单卡和四卡smoke的代码与cache hash必须匹配。`PYTHON_BIN`、`CACHE`可通过环境变量指定，默认解释器为robollm，默认缓存是`data/Planning/dual_object_a1_v1_20261004`。每组smoke包括8个真实对象丰富训练样本、反向与梯度检查、生成、保存及重新加载后的生成一致性；随机初始化阶段解析率低会记录，不拿smoke当规划成绩。

日志和产物：

- `T1_smoke.log`、`S1_smoke.log`及各自`smoke.json`：梯度/重载/生成证据。
- `T1_train.log`、`S1_train.log`：训练及validation进度。
- 每组`run.json`、`gradient_checks.json`、`history.jsonl`与`val_epoch_N.json`：配方、来源、逐epoch指标和全部预测文本。
- `epoch_N/`：语言LoRA、S1对象encoder、processor、接口metadata与optimizer/scheduler/RNG；`best_checkpoint.json`按coverage优先，再比较val平均L2选型；`complete.json`只在完整训练结束后生成。

## 数据与模型约定

共同缓存929train/157val/480test，训练运行器仅评测val；独立test入口见下方。独立原生GT→因果运动补偿→A1硬IoU关联→50m ROI→32个ego贡献对象与16个额外RSU对象。双端对象计一次且选择ego当前原始框，来源位[1,1]；未匹配RSU用补偿框。保留独立审计文件，输入无track/token/cooperative身份。

两组从同一缓存读取geometry13、source_bits和quality=[age_s,age_valid,velocity_valid]；对象速度输入双端统一禁用，因果速度只用于坐标补偿。文本采用完整float序列化；soft分支为共享13D MLP + xyz MLP + 三种source embedding + quality MLP + LayerNorm，每对象一个token。对象padding完全mask，无对象不产生伪token。两组相同的自车state/history仍为文本，原有ego perception对象文本不再另加。

视觉tower/projector冻结；在线语言前向训练LoRA(r/alpha8/8，dropout0)及S1 encoder，基础语言参数冻结。训练与生成统一提示token化，答案独立编码避免BPE边界变化；图像/对象/提示/padding label=-100。图像按本地冻结SigLIP和LLaVA projector生成729个tokens，在原图像占位处展开；soft对象在用户对象段替换占位，保留assistant边界。只在生成prefill拼接，后续直接Qwen KV cache解码。

6epoch、seed42、AdamW lr2e-4、weight_decay0、warmup3%、cosine、BF16、gradient checkpointing。答案CE按有效答案token计算，分块重计算避免全上下文词表logits占显存。所有输入禁止截断；train/val长度审计T1最大6521、S1最大1604，两组统一上限7168。未来GT九点只作为答案，指标用未舍入GT；严格JSON解析失败单独记录，coverage较低时不能只凭有效子集L2获胜。

缓存导出入口（使用新的目录）：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m tools.object_vlm.export --output-dir data/Planning/dual_object_a1_new
```

生成式validation入口（需要用户实际生成的checkpoint；新的输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zzn/anaconda3/envs/robollm/bin/python -m tools.object_vlm.run predict \
  --representation soft --checkpoint work_dirs/YOUR_RUN/S1_train/epoch_1 \
  --device cuda:0 --output-dir evaluation_results/S1_objects_val_new
```

`run train --checkpoint <epoch目录> --epochs 6`支持相同缓存/seed/预算的正式epoch恢复；另指定新输出目录。不会将smoke权重作为正式初始化。恢复输出中的best只覆盖恢复点与之后epoch，原运行更早checkpoint的选型记录仍需保留。

测试：

```bash
/home/zzn/anaconda3/envs/robollm/bin/python -m pytest tests/test_object_vlm.py \
  tests/test_relaxed_fusion.py tests/test_stable_fusion.py tests/test_native_association.py -q
```

旧continuous head、旧MMdrive双图像训练与旧数据保持原版本。RSU仅在VLM输入，没有planner旁路；不引入BEV、Map、StateEncoder、尺寸平均或动态范围。

2026-10-04四卡修订：torchrun四rank，显式同步可训练参数梯度；按全局batch4切分929条样本，最后一条仅rank0计算，其余rank仍参加同步，不重复或丢样本。validation四卡分片生成，rank0按原token顺序汇总；只有rank0写checkpoint/指标，恢复保存各rank RNG。四卡每组仍1398次optimizer更新，6epoch。13项本轮接口/分布式CPU测试通过，包括四进程Gloo梯度等价与最后不足batch场景。用户释放0–3卡后已明确授权运行脚本smoke；真实GPU/NCCL验证状态见最新实施记录。

启动脚本重复运行修复：train长度审计使用每次独立目录；审计错误直接打印，四卡训练输出通过tee同时显示终端并保存日志。训练输出目录已存在时明确报错，不覆盖。可以在原启动命令前加DRY_RUN=1，仅审计/检查smoke与输出路径，不启动GPU。中断后无epoch checkpoint需从头训练，保留旧T1_train目录后再启动。

## 完整 test 评测（2026-10-05）

已完成两组6epoch训练。运行：

```bash
EVAL_GPUS=0,1,2,3 bash tools/object_vlm/eval_t1_s1.sh
```

默认读取`work_dirs/T1_S1_objects_4gpu_smoke_20261004`，按validation coverage优先、平均L2次之冻结checkpoint（当前T1 epoch5、S1 epoch6）。顺序运行T1、S1，各使用四卡分片生成同一份480帧test，复用原validation的严格JSON解析、九步轨迹、确定性生成与指标实现；不会重新选epoch或训练。每卡加载完整模型，默认四卡减少生成时间；EVAL_GPUS也接受单卡或两卡。

输出到新的`evaluation_results/T1_S1_test_<时间>`：`frozen_selection.json`保存数据、权重和代码哈希；T1/S1各含`predictions.json`、`metrics.json`、`complete.json`；根目录`report.md`和`comparison.json`汇总L2@1/2/3s、平均L2、FDE@4.5s、解析率、输入tokens。终端显示加载与生成进度，T1.log/S1.log同步保存；权重加载阶段可能暂时无逐帧输出。旧输出不覆盖。可通过RUN_ROOT、CACHE、EVAL_ROOT、PYTHON_BIN指定路径。

仅检查输入与冻结checkpoint、不启动GPU：

```bash
DRY_RUN=1 bash tools/object_vlm/eval_t1_s1.sh
```

误差仅统计合法预测，解析失败单独报告；无碰撞评测器时collision=null。单seed、GT Oracle对象，T1/S1比较仅衡量对象表示变化，不能证明双端相对ego-only的收益。本轮仅提供脚本和CPU预检查，没有启动test推理。


2026-10-05 test长度超限修复：用户首轮test评测在T1单帧prefill 7180>7168时退出，尚无完整test成绩。原预检查漏审计test提示长度；7168为训练包装器guard，基础Qwen配置支持32768。evaluate.py现对完整480帧两种表示的观测提示进行CPU长度审计（不读轨迹答案），记录T1最大7180/1帧超guard、S1最大1242/0帧超guard；统一评测guard提升到8192，额外检查512生成预算在模型context内。checkpoint加载后显式覆盖model.max_length，原model.py及训练权重/代码保持不变；不截断、不删对象、不重新选epoch。修复NCCL销毁后异常hook调用get_rank的连带报错。4项评测CPU测试通过，最终预检查见evaluation_results/T1_S1_test_context_preflight_20261005/frozen_selection.json。用户仍用原命令启动，自动新输出目录，旧失败目录保留；本次未启动GPU。
