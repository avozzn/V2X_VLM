# I2视觉／空间queries：OmniDrive、ORION、VGGDrive代码核查与开发方案

更新：2026-10-05。范围是VLM基座、视觉／几何特征进入语言模型的路径，以及对当前I2的适配；不改变自车状态／历史、轨迹生成或已有T1/S1训练产物。本轮完成本地代码与原文核查和开发接口设计，尚未实现I2训练入口或启动GPU。

## 1. 核查结论与模型基座

| 项目 | 语言／视觉模型 | 进入语言模型的信息 | 与本项目关系 |
|---|---|---|---|
| OmniDrive本地版本 | `LlavaLlamaForCausalLM`，本地7B、hidden4096的Llama系语言模型；EVA-02-L视觉编码器；Q-Former-PETR | 感知decoder内专用carrier queries，经投影形成视觉语言tokens | 借鉴图像memory＋3D位置编码＋carrier读取；不是现有SigLIP投影权重的直接替换 |
| ORION | 论文为Vicuna v1.5＋EVA-02-L；本地类为`LlavaLlamaForCausalLM`，复用OmniDrive预训练路径 | 当前scene/carrier tokens＋历史queries，经MLP进入LLM；主规划用planning token接生成器 | 借鉴QT-Former和因果memory；当前不复用其规划器，不改轨迹文本输出 |
| VGGDrive | Qwen2.5-VL-7B＋冻结VGGT几何模型 | 保留VLM视觉tokens，在decoder层内以视觉hidden读取VGGT特征并残差注入 | 借鉴几何memory和跨注意力适配；不是固定少量query替换视觉tokens的同一种实现 |
| 当前T1 / I1-S1 | 本地LLaVA包装Qwen1.5-7B-Chat语言模型，SigLIP视觉模型＋原projector | 原始车端729视觉tokens＋对象文本／数值soft tokens＋state/history文本 | I2首轮沿用基座、来源、对象和语言输出合同 |

**模型身份的证据边界：**OmniDrive本地`ckpts/config.json`确认7B预训练路径、4096 hidden、32层、`LlavaLlamaForCausalLM`、`qformer-petr`、EVA-02和256 queries，但该配置没有写明Vicuna具体版本，不能仅从Llama类名推断聊天模型来源。ORION论文明确Vicuna v1.5；Qwen2-VL在其论文中用于生成Chat-B2D标注，不是ORION主语言模型。VGGDrive推理脚本是7B；某些Python入口默认3B，不能把未覆盖的默认值当正式实验基座。

证据：

- [OmniDrive本地权重配置](/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main/ckpts/config.json)、[主配置](/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py)。
- [ORION原文§4.2](https://arxiv.org/html/2503.19755v1#S4.SS2)、[加载器](/home/zzn/V2X_VLM/Orion-main/mmcv/utils/misc.py:413)、[stage3配置](/home/zzn/V2X_VLM/Orion-main/adzoo/orion/configs/orion_stage3_train.py:171)。
- [VGGDrive原文§3.1](https://arxiv.org/html/2602.20794v1#S3.SS1)、[NAVSIM推理脚本](/home/zzn/V2X_VLM/VGGDrive-main/run_scripts/inference_navsim.sh:21)。
- [当前LLaVA配置](../checkpoints/LLM/llava-next-interleave/config.json)。

## 2. OmniDrive：感知queries与carrier queries分工

```text
多相机图像 → EVA-02 → patch memory F
相机标定＋像素位置＋采样深度 → PETR 3D位置编码 E
                                ↓
检测queries＋carrier queries → self-attention
                                ↓
                  cross-attention读取(F＋E, F)
                     ├─ 感知queries → 检测／地图监督
                     └─ carrier queries → 投影到4096 → LLM → 文本
```

这里送给LLM的并非每个预测框对应的一条13D数值embedding，而是感知decoder中的专用carrier hidden。它们通过与感知queries交互、读取视觉memory学习场景表示。本地实现对对象和地图分支返回的carrier分别拼接，再以`images=vision_embeded`送入LLM；原始dense patch并未作为另一完整P块同步送入该语言路径。我们要求P＋Q是项目改造，不能称为原论文原样实现。

`position_embeding`将每个patch中心与多档深度构成相机射线采样点，乘`lidar2img.inverse()`得到LiDAR系候选3D位置，再归一化、编码。**采样深度不是实测LiDAR深度；3D位置编码也不是建立稠密BEV特征图。**本地配置`use_lidar=False`。

本地配置检测`num_query=600`、检测／地图各`num_extra=256`；论文v1提到检测900、地图300、carrier256。这是配置／版本差异，移植时以所用配置为准。

代码入口：

- [3D位置编码](/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main/projects/mmdet3d_plugin/models/detectors/petr3d.py:203)。
- [图像／感知／语言forward](/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main/projects/mmdet3d_plugin/models/detectors/petr3d.py:270)。
- [carrier初始化和提取](/home/zzn/V2X_VLM/RoboLLM/zoo/OmniDrive-main/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py:527)。
- [原文§2.2–2.4](https://arxiv.org/html/2405.01533v1#S2.SS2)：2D视觉语言预训练后3D驾驶适配。不能假设随机carrier在929帧上天然复现其效果。

## 3. ORION：在上述结构上增加历史读取

ORION的QT-Former包含当前scene queries、perception queries，以及少量history queries。当前queries读取带3D位置编码的图像memory；感知分支有对象、交通状态、运动等辅助任务。历史query先读取带相对时间编码的memory，再读取当前scene queries，更新后存入FIFO。scene/history hidden投影后进入LLM。

本地`orion_head.py`里的`memory_query`是16条；`memory_decoder_mq`读取历史scene memory，`memory_decoder_cq`读取当前`vlm_memory`。当前carrier `num_extra=256`，检测和地图两分支；不要把这项直接写成论文总scene512的单分支配置。代码还可加入can_bus/历史位姿embedding，与我们固定state/history文本的首轮协议不同。

主规划路径使用专用waypoint/planning token和连续生成器；本地stage3设置`use_gen_token=True`、`use_diff_decoder=False`，另有VAE相关loss和diffusion／MLP可选分支。I2只借鉴语言模型前的查询模块，不移植这些规划输出分支。

代码入口：

- [当前与历史query交互](/home/zzn/V2X_VLM/Orion-main/mmcv/models/dense_heads/orion_head.py:733)。
- [history读取与投影](/home/zzn/V2X_VLM/Orion-main/mmcv/models/dense_heads/orion_head.py:775)。
- [scene memory更新](/home/zzn/V2X_VLM/Orion-main/mmcv/models/dense_heads/orion_head.py:507)：使用detach存储，不代表当前可训练query输出可以在进入LLM前detach。
- [VLM与规划路径](/home/zzn/V2X_VLM/Orion-main/mmcv/models/detectors/orion.py:565)。
- [原文§3.1–3.3](https://arxiv.org/html/2503.19755v1#S3.SS1)。

## 4. VGGDrive：保留视觉tokens，在层内读取几何memory

```text
同一组多视图图像 ── Qwen2.5-VL视觉编码器 → 原视觉tokens
           └──── 冻结VGGT → 跨视图几何memory
                                     ↓
LLM每层中的视觉hidden → 低维投影 → cross-attention读取VGGT
                                     ↓
                    投影回语言维度＋残差 → 下一层
```

本地NAVSIM入口使用`CustomQwen2_5_VLForConditionalGeneration`。`run_vggt_inference`调用VGGT aggregator并取最终tokens，输出维度2048；每层将语言视觉hidden和VGGT特征分别降维到512，cross-attention融合，投影回语言hidden。cam版本将4×4 `img2lidar`展平后编码，加入对应视图首token。更新的仅是image mask对应的hidden，文字位置不会直接被覆盖。

训练时及生成prefill执行3D注入，已有KV缓存后的逐token解码不重复提取。代码是每层独立适配模块；不能把一次输入拼接称为其分层CVGE。它读取的是VGGT预训练几何特征，并非将GT框数值换一个MLP名称。

论文采用adapter预热后联合微调，VGGT保持冻结；这是训练启发，不直接照搬其数据和GPU配方。当前本地规定路径`vggt/model.pt`不存在，尚不能直接运行该分支；本项目车端前视＋路端高位异步双视图，也不同于原实验同车周视相机，需要另验证重叠、动态目标、时序与预训练域适配。

代码入口：

- [每层适配器](/home/zzn/V2X_VLM/VGGDrive-main/inject_utils/Qwen2_5_vggt_fusion_inject_cam.py:95)。
- [prefill／层内注入](/home/zzn/V2X_VLM/VGGDrive-main/inject_utils/Qwen2_5_vggt_fusion_inject_cam.py:196)。
- [冻结VGGT特征](/home/zzn/V2X_VLM/VGGDrive-main/inject_utils/vggt_utils.py:9)。
- [图像缩放与标定](/home/zzn/V2X_VLM/VGGDrive-main/inject_utils/utils.py:50)。
- [原文§3.2–3.3及§4.1](https://arxiv.org/html/2602.20794v1#S3.SS2)。

## 5. 对本项目I2的具体开发方案

下面是项目设计建议，不是上述论文的已有实现。先实现I2-A，验收后再扩展双端视觉与VGGT；状态编码和轨迹预测由用户另行开发，当前接口保持文本状态＋语言生成。

### 5.1 I2-A：同输入的对象条件视觉读取

```text
车端图像 → 现有冻结SigLIP → patch memory F ── 原projector → P（保留）
                                     │
独立双端GT → 原因果补偿／A1关联／ROI → geometry/source/quality
                                     ↓
                         对象条件query种子 q_i
                                     ↓
               cross-attention：query q_i，key F＋位置，value F
                                     ↓
                可见对象：几何种子＋受mask控制的视觉残差
                无视觉支持对象：保留几何种子，不编造图像特征
                                     ↓
                         投影到语言维度 → Q
                                     ↓
             P＋Q＋原state/history文本＋提示 → 同一VLM → 轨迹文本
```

首版一对象一query，沿用32＋16对象预算，不从256随机scene queries起步。对象种子编码13D几何、xyz、source与quality；读取维度先256，2层、8头作为工程默认值，后续只用val调整。SigLIP原始patch memory与原projector输出分别用于Q读取与P，避免重复跑视觉tower。

区别于I1：I1对象token只来自数值；I2-A对象query必须实际读取图像memory，其输出对图像变化可测。没有读取memory的geometry-only输出只能叫I1／几何原型。

**位置分两层实现并分别标注：**

1. 对象xyz条件＋图像2D patch位置／投影门控是轻量原型，只能称对象条件视觉查询，不能宣称已做PETR的3D patch编码。
2. 校准验收后，在patch key上加入射线深度采样3D编码，参考OmniDrive；记录深度档位、覆盖范围与位置归一化。此时也不等于实测深度或BEV。

视锥／投影只证明几何可投影，不能证明未遮挡。RSU-only或车端视野外对象不强制绑定车端patch；关闭其对象局部视觉读取后保留几何分支。可用背景scene查询另做独立实验，不把场景全局读取冒充该对象的真实外观。

P和Q包含重复车端视觉信息，须在方法和预算中显式披露。主组将S_i通过query模块增强后替换成Q_i，不再把同一批S_i完整追加一次。P＋S＋独立scene-Q属于另一组，并非与上述一对象一query相同的消融。

### 5.2 I2-B：路端也提取视觉queries

只有I2-A接口验收后再实现：路端用合法选中消息对应的图像、原生观测／标定，在本端读取patch特征形成queries，将几何位置因果补偿并转到当前ego系后传输。车端保留P、接收路端queries，再经VLM生成。

这一组增加了路端图像语义来源，不能只与geometry-only I1比较并把所有提升归于query架构。需增加同双端视觉来源的对照，分别说明原图像是否进入VLM、特征是否在路端编码、是否真实计算传输bytes。路端图像时间、标注时间与消息截止时间独立审计；不能拿当前ego时刻的补偿框直接投影到旧路端图像。

双端对象关联仍保持A1，不在I2-B同时引入学习式车路匹配／融合；后者留I3。两端无重叠也能各自提取本端queries；跨视域对应关系不应靠VGGT自动保证。

### 5.3 VGGDrive启发的几何memory扩展

作为I2-G后续分支：先验证冻结VGGT在本项目输入上的可用性，再令对象／scene queries读取VGGT memory。若选择每层视觉hidden注入，另命名I2-H，独立列参数、训练和推理开销，不能同一次输入级P＋Q混记。

先保留现有LLaVA基座；换Qwen2.5-VL必须额外跑对应基座的I1对照，否则无法区分基座升级和query收益。VGGT权重缺失与非重叠／异步输入是实质依赖，不以下载整个新模型替代首轮轻量接口验证。

### 5.4 实现文件与验收顺序

新增隔离的`tools/object_vlm_i2/`，不修改已有T1/S1运行时代码、缓存或checkpoint（原评测严格检查代码hash）。预期文件职责：

| 文件／步骤 | 要实现的内容 | 验收证据 |
|---|---|---|
| `geometry.py` | 标定、预处理resize/crop/pad变换、物理坐标→camera→patch映射、射线3D编码 | 已知点／框投影、前后轴、无效深度、图像边界、真实样本叠加检查 |
| `queries.py` | 对象种子＋cross-attention＋视觉支持mask＋残差＋输出投影 | 改图像memory时有效Q变化；padding不影响；不可投影对象保留几何；全mask无NaN |
| `model.py` | 一次SigLIP提取同时产生P与memory，Q在prefill插入，维持答案mask与KV | 梯度到query／LoRA、视觉冻结、生成只拼一次、保存重载一致 |
| `data.py` / `export.py` | 在新manifest中记录标定／原图尺寸／预处理及来源；不更换对象列表和target | 与I1逐帧对象一致；无future观测；消息时间和缺失mask明确 |
| `run.py` / `run_i2.sh` | CPU审计、单卡smoke、四卡probe、正式训练／评测入口 | 用户自行启动；提供日志；完整test预检查输入长度＋512生成预算 |

训练阶段要经过在线语言forward，不能将Q detach后训练新adapter。冻结视觉tower可以no_grad，缓存冻结视觉特征也可以，但query和语言LoRA的可训练路径要保持。首轮不添加目标未来运动监督、不同时改变状态接口和轨迹loss。

开发顺序：标定与mask → query模块CPU验收 → 真实7B接口smoke → 与I1同预算正式训练。模块预热如启用，单独记训练步数；不能给I2免费增加预算后称与现有6epoch I1严格同配方。历史memory借鉴ORION，排在单帧输入稳定后，按scene和timestamp管理并在scene切换清空，不随随机loader跨样本累积。

## 6. 最少实验与报告边界

| 实验 | 输入／变化 | 回答的问题 |
|---|---|---|
| I1 / S1已有 | P＋几何对象S＋state/history文本 | 当前基线；test主平均0.7158m |
| I2-A主组 | P＋对象条件视觉Q＋相同state/history文本；同一图像／对象信息来源 | 读取视觉语义是否优于仅几何编码？ |
| I2-A无视觉更新检查 | 屏蔽视觉残差；同一已训练权重仅作诊断 | 是否依赖图像memory？不能当独立训练成绩 |
| I2-A Q-only | 同Q移除P，独立训练，其他合同相同 | query能否承担压缩；背景／不可见对象损失多大？主组通过后再做 |
| I2-B／I2-G／I2-H | 分别新增路端视觉、VGGT或层内注入 | 后续专项，新增数据／预训练／预算必须披露 |

固定val选型，不用已查看的test调query数或几何门限。报告原主平均L2@1/2/3s，以及2.5/3.5/4.5s、4s、FDE4.5、解析率、tokens、真正端到端latency与allocated峰值显存；没有checker的collision/off-road仍为未评测。第一轮I2收益不自动等于协同收益，ego-only与RSU有无需另做同构对照。

## 7. 核查范围与限制

采用academic-research-suite的来源核查思路，直接核对三份原文的方法章节和本地配置／forward。未运行这三个仓库的模型或复现实验，不保证其他配置路径与当前审计入口相同。本地OmniDrive是较早版本，原文与配置query数差异已明确；ORION无本地ckpts配置可确认更细模型revision；VGGT指定权重未就位。没有用综述或第三方榜单替代主来源，也没有依据论文收益承诺当前929帧训练数据上的效果。
