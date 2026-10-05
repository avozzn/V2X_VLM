# 当前 MMdrive_v2x.py 训练能否作为 T1

核查日期：2026-10-04；只读宿主机进程、当前源码和JSON、已保存trainer_state。未改动或停止训练。

结论：不能作为独立双端GT对象文本T1，可保留为双图像+自车状态/历史文本的旧基线。

## 实际运行与进度

- bash父进程PID3104004；torchrun PID3104007；4个主训练rank PID3104113～3104116，读进程时已运行约2小时18分钟。
- 实际config `projects/configs/Robodrivevlm/MMdrive_v2x.py`，输出 `checkpoints/V2X_dt_9step_4096`。
- 6epochs、per-device batch1、grad accumulation1、语言LoRA r/alpha=8/8、lr2e-4、seed42、eval_strategy=no；视觉encoder/projector冻结。config的MMEngine max_epochs24不是实际HF命令的6epochs。
- checkpoint-368：epoch1，最后loss0.3873；checkpoint-736：epoch2，最后loss0.4524。这是保存快照，不是实时step，也不是validation规划成绩。

## T1不符合之处

1. `train_pipeline`的LoadMultiViewImageFromFiles4Clip设置only_vehicle=False，提供双图像；拟议T1/S1首版只提供车端原图，RSU通过独立对象表示输入。
2. `data/Planning/train_v2x_causal_history.json`为1470条，`test_v2x_causal_history.json`为676条；所有human文本都没有Perception段，也没有agents/ego_perception结构。示例包含自车尺寸、速度、加速度、4点历史及mask。
3. dataset `process_json_file`将JSON human/assistant写入QA_pairs，Load_EGO拼接QA_pairs[0]，Load_planning_prompt拼接输出提示并将QA_pairs[1]作为训练答案。prepare_data中动态generate_full_prompt调用被注释，文件底部generate_user_message对象formatter未沿此路径调用。
4. 当前pipeline未读取原生独立vehicle-side/infrastructure-side GT并转为两源对象文本。加载cooperative PKL本身不能证明独立双端对象进入语言输入。
5. 数据条数与当前continuous_v2_ego_gt 929/157/480不同；当前输出合同legacy。对T1/S1须冻结同split/相同轨迹模板，不能直接对比分数归因于soft tokens。

当前JSON完整内容与运行进程参数相符；本轮没有保存启动时source snapshot，若运行中有人改过源码/JSON，最终有效输入应以当时记录或实际batch为准。上述身份按当前可读文件确认，不能凭配置文件名推断其他输入。

## 最小实验决定

只做T1/S1，共享独立双端GT对象集合、当前ego坐标、范围/预算、时间配对、缺失规则、车端图像、自车状态历史表示、LLM/LoRA与输出评测。先固定state/history文本以仅比较对象表示，StateEncoder另排；若先实现StateEncoder则两组同步使用。这个最小实验不能量化路侧相对ego-only的收益，也不把StateEncoder记为已实现。

本轮未实现soft-token语言训练/generate或启动新训练；当前训练保持运行，已有产物保留。
