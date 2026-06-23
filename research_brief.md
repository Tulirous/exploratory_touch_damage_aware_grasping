# 初始研究备忘录

日期：2026-05-13

## 1. 你现在真正要做的不是“大机器人课题”

你之前接触的是人体三维重建。这个背景并不浪费，反而有几个优势：

- 你熟悉从传感数据中恢复结构或状态。
- 你熟悉监督学习、误差定义、数据标注和泛化。
- 你更容易接受“中间表示”的价值。

但机器人课题和三维重建最大的差异是：机器人系统必须闭环执行。也就是说，论文不能只证明“估计得准”，还要证明“估计结果让动作变好”。

本课题的关键不是重建几何，而是从早期触觉中估计对动作有用的物理属性：

```text
触觉信号 -> 抓取相关物理状态 -> 抓取参数 -> 成功且少损伤
```

进一步地，本课题可以表述为一个轻量级 World-Action Model, WAM 问题：

```text
早期视觉-触觉状态 + 候选抓取动作
  -> 未来接触状态 / 滑移风险 / 损伤风险
  -> 安全抓取参数
```

这里的 WAM 不是从零预训练 DreamZero、UWM 或通用机器人基础模型，而是面向超柔软易损物体的任务特化 WAM。当前路线允许直接使用 Fast-WAM / Wan / WALL / OpenPI 这类预训练 backbone，但本课题自己的核心模块是触觉-力觉 adapter 和候选动作后果预测。它的核心不是推理时生成未来视频，而是预测不同抓取参数会造成什么后果，例如是否滑移、是否压坏、形变是否可恢复。

## 2. 现有相关工作的几条主线

### 2.1 柔软/易损物体操作

这类工作把 fragile object manipulation 从普通抓取中分离出来。核心观点是：对易损物体，目标不是单纯完成 pick-and-place，而是在满足安全约束下完成操作。

需要关注的关键词：

- deformable object manipulation
- fragile object manipulation
- damage-aware grasping
- safety-aware manipulation
- adaptive grasp control

代表性综述指出，易损物体操作的关键缺口是：现有方法经常没有把 fragility constraint 作为核心设计目标，而是事后通过力阈值或软夹爪被动解决。

对本课题的启发：

```text
你的论文必须显式定义 damage risk，而不是只报告抓取成功率。
```

### 2.2 单次触觉 / 首次接触估计物理属性

已有工作证明，机器人可以从 single grasp 或 first touch 中估计物体属性，例如 compliance、Young's modulus、pose 或接触稳定性。

代表方向：

- single-grasp compliance estimation
- first-touch pose estimation
- tactile probing
- stiffness / hardness estimation

对本课题的启发：

```text
只做“估计柔软度”已经不够。更好的贡献是：估计结果能生成更安全的抓取参数。
```

### 2.3 低成本触觉

很多高水平触觉工作依赖 GelSight、TacTip 或 optical tactile sensor。你的压力垫更廉价，也更粗糙。

这既是缺点，也是切入点。

缺点：

- 漂移
- 迟滞
- 非线性
- 空间分辨率有限
- 难以估计精细几何

可转化成的贡献：

```text
低成本电阻式触觉是否足以支撑早期损伤风险预测和安全抓取？
```

### 2.4 VLA / 具身智能

VLA 的典型形式是：

```text
视觉 + 语言指令 -> 动作
```

近期 tactile-VLA 工作进一步加入触觉：

```text
视觉 + 语言 + 触觉 -> 动作
```

代表方向包括 TLA、VTLA、VLA-Touch、HapticVLA 等。它们通常聚焦 contact-rich manipulation，比如 peg-in-hole、insertion、disassembly 或一般 manipulation。

你的课题和 VLA 的关系应谨慎定位：

- 第一篇主线不需要是 VLA。
- 你可以把它称为 embodied manipulation 或 tactile embodied intelligence。
- 后续可以把你的触觉模块接到 base VLA 上，做 contact-phase safety refinement。

### 2.5 WAM / 世界-动作模型

WAM 的典型思想是同时考虑：

```text
world prediction: 这样做会发生什么？
action generation: 应该怎么做？
```

对本课题来说，WAM 的落点非常自然：

```text
轻触 / 轻夹得到早期状态
候选抓取参数作为 action
抓取后的滑移、损伤、形变作为 future state
模型学习 state-action -> future outcome
```

这比只估计 hardness 或 compliance 更进一步，因为模型直接学习“某个动作会不会造成损伤”。第一篇不建议从零训练通用 WAM，而应做 Fast-WAM-style 视触觉后果预测：

- 视觉 / 视频 latent：来自冻结的 Fast-WAM / Wan / WALL / OpenPI backbone，或 RGB-D 特征编码器
- 触觉-力觉 latent：轻触阶段的压力序列、夹持力曲线、夹爪开度、闭合速度
- 动作 latent：候选夹持力、闭合距离、闭合速度、抬升速度
- 未来后果：抓取成功、滑移、损伤、压缩形变、恢复率

第一版建议采用：

```text
visual_latent + tactile_force_latent + candidate_action_latent -> future_outcome
```

也就是对多个候选抓取动作进行后果预测和风险打分。推理时不生成未来视频。稳定后再升级为：

```text
state -> future_contact_state + grasp_action
```

这才更接近 Joint WAM。

## 3. 具身智能和本课题的关系

具身智能不是“用了大模型”才算。更基本的定义是：

```text
智能体通过身体与环境交互，利用感知-行动闭环完成任务。
```

按这个定义，本课题本身就是具身智能的一类：

- 机器人主动轻触环境
- 从接触反馈中估计物体状态
- 根据估计结果调整动作
- 通过真实物理交互验证策略

本课题与 VLA 的区别：

| 概念 | 重点 | 本课题位置 |
|---|---|---|
| 具身智能 | 感知-行动闭环、物理交互、环境反馈 | 本课题天然属于具身智能 |
| VLA | 视觉、语言和动作的大模型或策略模型 | 可作为后续扩展 |
| tactile-VLA | 在 VLA 中加入触觉 | 可作为第二篇或升级版 |
| 本课题第一阶段 | 触觉试探、损伤风险、抓取参数生成 | 更偏机器人触觉控制和安全抓取 |

所以你不需要为了“具身智能”强行上 VLA。只要完成真实机器人上的主动触觉试探和闭环抓取，这已经是 embodied manipulation。

## 4. 两条递进式研究路线

当前课题可以拆成两条路线，但它们不是并列同时推进，而是前后递进：

```text
路线一：易损物体抓取
  -> 建立触觉采集、损伤标签、后果预测、安全抓取参数生成

路线二：人机递交中的损伤/安全感知抓取
  -> 在路线一基础上加入人手状态、安全区域、递交时序和释放策略
```

### 4.1 路线一：易损物体抓取

路线一是当前最应该先完成的主线。它的对象是豆腐、果冻、海绵、软糕点等超柔软易损物体，任务是让机器人在正式抓取前先轻触或轻夹，通过早期触觉和 RGB-D 物体状态预测候选抓取动作的后果。

核心科学问题：

```text
机器人能否通过一次轻触，预测某个抓取参数是否会导致滑移或损伤？
```

输入：

- RGB-D 估计的物体尺寸、形状、候选接触区域
- 早期压力垫序列
- 夹持力曲线
- 夹爪开度、位移、闭合速度
- 候选抓取参数，例如夹持力、闭合距离、闭合速度、抬升速度

输出：

- 抓取成功概率
- 滑移风险
- 损伤风险
- 压缩形变和恢复率
- 推荐安全抓取参数

对应模型：

```text
object_state + tactile_state + candidate_grasp_action
  -> future_outcome
```

这条路线的优势是变量可控、实验边界清晰、数据采集难度较低，适合作为第一篇论文。它可以先不涉及人、不涉及语言、不涉及大规模 VLA，只要证明“轻触后果预测能够降低损伤并提高抓取成功率”即可。

### 4.2 路线二：人机递交中的损伤/安全感知抓取

路线二是路线一的扩展，不建议一开始直接做。它把任务从“抓起易损物体”扩展为“安全地把易损物体递交给人”。

新增问题不是单纯多了一个人，而是安全约束从单目标变成多目标：

```text
不要压坏物体
不要夹到人
不要在人还没接稳时释放
不要在人拉扯时继续强行夹持
```

新增输入：

- RGB-D 中的人手位置、姿态、接近方向
- 人手机械臂之间的安全距离
- 递交区域或安全区域
- 人是否接触物体的视觉/触觉/力觉证据
- 释放前后的夹持力变化

新增输出：

- 人手接触风险
- 交接成功概率
- 是否允许释放
- 是否需要暂停、后退或重新递交
- 递交动作参数，例如递交姿态、速度、释放时机

对应模型：

```text
object_state + human_hand_state + tactile_force_state + candidate_handover_action
  -> object_damage_risk + human_safety_risk + handover_success
```

路线二可以复用路线一的三个核心资产：

- 触觉和夹持力数据采集流程
- 物体损伤/滑移/形变风险预测模型
- 候选动作后果预测和风险打分框架

路线二新增的是人机交互模块，而不是推翻路线一重做。具体可以先从静态递交开始：人手伸到固定区域，机器人慢速递交，检测到人接稳后释放。稳定后再加入自然手部运动、语言指令或更复杂的老人辅助场景。

### 4.3 推荐论文/项目顺序

推荐顺序：

```text
第一阶段 / 第一篇：
Exploratory Touch for Damage-Aware Grasp Synthesis of Ultra-Soft Fragile Objects
面向超柔软易损物体的探索性触觉与损伤感知抓取参数生成

第二阶段 / 第二篇：
Vision-Tactile Safety-Aware Grasping and Handover for Fragile Object Assistance
面向易损物体辅助递交的视触觉安全感知抓取与释放
```

为什么先做路线一：

- 不涉及真人交互，安全和伦理压力小。
- 损伤标签更容易定义和重复采集。
- 可以先把触觉、力觉、RGB-D 和候选动作后果预测跑通。
- 容易做 baseline：固定抓取、无轻触、压力阈值、只估计柔软度、WAM-style 后果预测。
- 做成后，路线二可以直接继承模型框架和数据协议。

为什么路线二适合作为后续扩展：

- 人机递交更贴近“照顾老年人”的应用场景。
- 机器人不仅要会抓，还要知道何时递、何时放、何时停止。
- 该方向可以自然引入人手状态、安全区域、语言指令和老人辅助任务。
- 如果路线一已经证明“损伤感知抓取”有效，路线二的新贡献就可以集中在人机交互安全和递交流程上。

## 5. 初步工作流程

### Step 1：确定物体族

第一阶段建议只做超柔软块体类物体：

- 豆腐
- 果冻
- 海绵
- 软糕点

暂时不要加入纸巾。纸巾是 thin-sheet manipulation，机制不同。

### Step 2：搭建轻触采集流程

每次 trial 包含：

1. 机械臂移动到预设位置。
2. 夹爪以固定速度闭合。
3. 压力垫检测到接触。
4. 执行小幅压缩。
5. 记录压力时间序列、夹爪位移、闭合速度。
6. 停止或进入正式抓取。

### Step 3：建立外部近似标签

至少需要三类标签：

- 是否成功抓起
- 是否滑移
- 是否损伤

更强的标签：

- 压缩曲线
- 永久形变
- 高度恢复率
- 表面破裂
- 渗水或质量损失
- 安全压缩范围

### Step 4：第一版模型

当前选择 Fast-WAM-style 大模型路线，但不要从零训练 Wan2.2-5B 或通用 WAM。第一版建议：

```text
冻结视觉/视频 backbone latent + 触觉-力觉序列 + 候选抓取参数
  -> tactile-force adapter / action encoder / DiT-style fusion
  -> future slip / damage / deformation risk
  -> 选择低风险抓取参数
```

候选 backbone：

- `Fast-WAM / Wan2.2-5B visual-video backbone`
- `WALL-OSS-0.5`
- `OpenPI`
- `RGB-D encoder`，作为工程保底

推荐第一版工程：

```text
Frozen visual/video backbone
  + Tactile-force adapter
  + Candidate action encoder
  + DiT-style fusion transformer
  + Multi-head outcome predictor
```

输出头：

- compliance regression head
- damage-risk classification head
- slip-risk classification head
- deformation / recovery regression head
- grasp-success classification head
- grasp-parameter regression head

### Step 5：抓取参数生成

第一版不要端到端输出完整机械臂轨迹。只输出低维参数：

- 目标闭合位移
- 目标抓力，如果可控
- 闭合速度
- 抬升速度
- 是否重新轻触

这样更容易训练和解释。

更推荐的第一版不是直接生成唯一动作，而是生成多个候选抓取参数，再让轻量 WAM 预测后果：

```text
候选参数 1 -> success / slip / damage score
候选参数 2 -> success / slip / damage score
候选参数 3 -> success / slip / damage score
```

最终选择：

```text
argmax(success - slip_risk - damage_risk - deformation_penalty)
```

### Step 6：闭环验证

核心对比：

- 固定抓取参数
- 无轻触直接抓取
- 原始压力阈值控制
- 只估计 hardness 的方法
- 本方法：轻触状态 + 候选动作后果预测 + 安全抓取参数筛选

## 6. 推荐网络模块

### 第一阶段：Fast-WAM-style latent outcome model

输入：

```text
visual_latent + pressure[t, c] + force[t] + gripper_position[t] + candidate_action[h, a]
```

如果压力垫是单通道：

```text
pressure[t] + displacement[t]
```

如果压力垫是阵列：

```text
pressure[t, h, w]
```

第一版模型：

```text
Frozen visual/video latent
  + tactile-force latent
  + candidate action latent
  -> DiT-style Fusion Transformer
  -> Multi-Head Future Predictor
```

当前工程骨架：

- `configs/fastwam_fragile.yaml`
- `datasets/fragile_episode_dataset.py`
- `models/tactile_force_adapter.py`
- `models/action_encoder.py`
- `models/outcome_transformer.py`
- `models/fastwam_fragile.py`
- `train.py`

第一阶段不要做：

- 从零训练 Wan2.2-5B video DiT
- 从零训练通用 VLA
- 全量微调大视频模型
- 把未来视频生成作为必要推理步骤

可以做：

- 冻结 Fast-WAM/Wan/WALL/OpenPI backbone
- 预计算 visual latents
- 训练 tactile-force adapter、action encoder 和 outcome head
- 数据稳定后做 LoRA / adapter 微调

### 第二阶段：策略学习

当你已经有稳定数据后，可以试：

- Behavior Cloning
- Diffusion Policy
- ACT
- Residual Policy

但动作空间仍建议低维化：

```text
delta_closure, delta_speed, lift_speed, retry_flag
```

### 第三阶段：VLA 扩展

如果要和 VLA 连接，建议结构是：

```text
Base VLA proposes action
Tactile module observes early contact
Tactile residual corrects contact-phase action
```

不要从零训练一个 VLA。

### 第四阶段：Fast-WAM backbone 接入

Fast-WAM 的 Wan2.2-5B video DiT backbone 应作为冻结特征提取器或 LoRA 微调对象，而不是复现目标。更具体地：

```text
RGB/RGB-D/video
  -> frozen Wan/Fast-WAM visual backbone
  -> visual_latent
  -> FastWAM-Fragile outcome model
```

如果 Fast-WAM 工程不稳定，则保底选择是 WALL-OSS-0.5 或 OpenPI 作为动作/视觉 backbone。

## 7. 数据怎么收集

### 数据单元

每条数据至少包含：

- object_id
- object_type
- trial_id
- LeRobot `observation.images.base`，D435 全局相机视频
- LeRobot `observation.images.wrist`，D415 腕部相机视频
- gripper pose
- gripper displacement sequence
- pressure sequence
- chosen grasp parameters
- lift result
- damage label
- slip label

### 推荐采集规模

快速原型：

- 4 类物体
- 每类 5 个样本
- 每个样本 20-30 次 trial
- 总计约 400-600 条 trial

较完整论文：

- 6-8 类物体
- 每类多个硬度或含水率版本
- 总计 1000+ 条 trial

### 采集策略

当前可以先用 GELLO 遥操作采集 LeRobot V3.0 episode，再补充本课题标签和 sidecar 触觉/力觉文件：

```text
GELLO + UR5 + D415/D435
  -> LeRobot episode
  -> pressure/force/gripper sidecar
  -> damage/slip/deformation labels
  -> FastWAM-Fragile manifest
```

GELLO 采集规范见：

- `GELLO 使用教程指南.txt`
- [docs/gello_to_fastwam_pipeline.md](docs/gello_to_fastwam_pipeline.md)

在模型训练前，建议从遥操作数据中生成：

```text
data/manifests/train.jsonl
data/manifests/val.jsonl
data/latents/<episode_id>_visual.npy
```

后续如果要减少遥操作偏差，再用脚本采集：

- 固定接近位姿
- 固定轻触速度
- 随机轻触压缩量
- 随机正式抓取参数
- 记录结果

这样数据更均匀，也更容易做回归。

## 8. 训练方式

### 训练任务 1：早期触觉到物理属性

目标：

- compliance regression
- safe compression range regression
- slip risk prediction
- damage risk prediction

损失函数：

```text
L = L_compliance + L_safe_range + L_slip + L_damage
```

### 训练任务 2：早期触觉到抓取参数

目标：

- 预测低维抓取参数
- 或预测参数是否安全
- 或对候选抓取参数预测未来抓取后果

三种路线：

1. 先预测物理属性，再用规则生成抓取参数。
2. 直接从触觉预测抓取参数。
3. 输入候选抓取参数，预测该参数的 success / slip / damage / deformation。

第一篇更推荐路线 3，因为它最符合轻量 WAM 表述，也更容易做候选动作筛选和 baseline 对比。

### 训练任务 3：策略残差

后续可以训练：

```text
base grasp parameters + tactile features -> residual correction
```

这条路线最容易和 VLA 连接。

### 训练任务 4：轻量 WAM 后果预测

目标：

```text
state_t + action_candidate -> future_outcome
```

输入：

- RGB-D 物体状态或视觉特征
- 轻触阶段触觉 / 力觉序列
- 夹爪位移、速度、开度
- 候选正式抓取参数

输出：

- grasp success probability
- slip risk
- damage risk
- deformation / recovery regression
- safe force or safe compression margin

推荐损失：

```text
L = L_success + L_slip + L_damage + L_deformation + L_safe_margin
```

推理：

```text
枚举或采样多个候选抓取参数
  -> 逐个预测 future_outcome
  -> 选择综合风险最低的参数
```

## 9. 初始实验设计

### 实验 A：早期触觉是否能预测损伤风险

问题：

```text
只看轻触前 0.5-2 秒的压力信号，能不能预测后续抓取是否会损伤？
```

### 实验 B：轻触是否提升抓取效果

比较：

- no touch
- raw pressure threshold
- predicted compliance only
- predicted damage + slip risk
- WAM-style candidate action outcome prediction

### 实验 C：未见物体泛化

训练：

- 豆腐 A、果冻 A、海绵 A

测试：

- 新豆腐品牌
- 新硬度果冻
- 新软糕点

### 实验 D：与具身智能/VLA连接

后续实验：

- 语言指令：“轻轻抓起，不要压坏”
- base policy 给初始动作
- tactile module 修正接触阶段动作

### 实验 E：WAM 后果预测是否有效

问题：

```text
模型是否能在正式抓取前，预测不同候选参数的未来损伤/滑移结果？
```

比较：

- 只用视觉状态预测后果
- 只用触觉轻触序列预测后果
- 视觉 + 触觉，不输入候选动作
- 视觉 + 触觉 + 候选动作，也就是轻量 WAM

关键指标：

- damage prediction AUC / F1
- slip prediction AUC / F1
- deformation regression error
- chosen action 的真实损伤率和成功率

### 实验 F：人机递交安全扩展

这是路线二实验，建议在路线一闭环稳定后再做。

问题：

```text
路线一学到的损伤感知抓取模型，能否扩展到人机递交中的安全释放和接触风险控制？
```

第一版递交实验可以先做受控版本：

- 人手进入固定递交区域。
- 机器人用路线一生成的安全抓取参数拿起物体。
- RGB-D 检测人手是否接近和是否处于安全区域。
- 夹持力/压力变化判断人是否接稳。
- 满足条件后释放，否则保持、暂停或后退。

核心对比：

- 固定释放时机
- 只用视觉判断人手是否到位
- 只用力阈值判断是否释放
- 本方法：物体损伤风险 + 人手安全区域 + 力/触觉接稳判断

指标：

- handover success rate
- premature release rate
- excessive contact force rate
- object damage rate
- handover completion time

## 10. 当前最该补的基础

你需要补的不是“全部机器人学”，而是这几块：

1. 机械臂坐标系和末端位姿
2. 夹爪控制：位置控制、速度控制、力/电流近似控制
3. 触觉传感器标定：漂移、迟滞、饱和
4. 抓取基本概念：法向力、摩擦、滑移、接触面积
5. 机器人学习基本概念：behavior cloning、policy、action space
6. VLA 基本概念：vision-language-action、action token、fine-tuning、embodiment gap

## 11. 我现在需要你确认的问题

这些问题会决定第一版方案怎么落地：

1. 你的夹爪是否能读到实际夹持力，还是只能控制位置/速度？
2. 压力垫是单点、条状，还是二维阵列？
3. 压力垫采样频率大概是多少？
4. UR5 当前是否已经能通过 Python/ROS 控制？
5. 实验室有没有 RGB 或 RGB-D 相机？
6. 你更希望第一篇偏机器人控制、触觉感知，还是偏具身智能/VLA？
7. 你是否能接受第一篇先不使用灵巧手，只用平行夹爪把主线做扎实？

## 12. 当前阶段的工作原则

- 第一阶段只做一个物体族，不做所有软物体。
- 第一版只输出低维抓取参数，不输出完整轨迹。
- 第一版使用预训练大模型 backbone，但不从零训练大模型。
- 必须定义 damage metric。
- 必须做 unseen object 测试。
- VLA / WAM backbone 是视觉-动作先验，课题核心是触觉-力觉后果预测和真实闭环验证。
- 先完成易损物体抓取，再扩展到人机递交；不要一开始同时做两个闭环。
