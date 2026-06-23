# Exploratory Touch for Damage-Aware Grasp Synthesis

项目日期：2026-05-13

## 项目定位

工作题目：

```text
Exploratory Touch for Damage-Aware Grasp Synthesis of Ultra-Soft Fragile Objects
面向超柔软易损物体的探索性触觉与损伤感知抓取参数生成
```

核心问题：

机器人在完整抓取之前，先通过一次轻触或轻夹获取早期触觉信号，估计物体的抓取相关物理属性，再生成更安全的抓取参数，降低滑移和损伤。

当前路线已经调整为：第一篇不从零训练 VLA 或通用视频 WAM，但可以直接使用预训练 Fast-WAM / Wan / WALL / OpenPI 这类大模型 backbone 作为视觉-动作先验，再围绕本课题设计触觉-力觉 adapter 和 WAM outcome head。

```text
早期视觉-触觉状态 + 候选抓取动作
  -> 未来接触状态 / 滑移风险 / 损伤风险
  -> 安全抓取参数
```

这里的 WAM 不是从零预训练大规模通用机器人基础模型，而是面向超柔软易损物体的任务特化后果预测模型。它的核心价值是：机器人在正式抓取前预测不同抓取参数可能造成的后果，再选择低损伤、低滑移的动作。推理时不要求生成未来视频。

## 固定硬件与软件环境

以下环境作为项目硬约束，后续实现默认不更改：

- 操作系统：Ubuntu 24.04.4 LTS
- 环境管理：Conda
- 本地算力：RTX 4070 8GB 笔记本，负责数据采集、设备控制和轻量闭环推理客户端
- 云端算力：按需租用 GPU 服务器
  - 最低配置：24GB 显存，用于 outcome head / adapter 训练、小批量 latent extraction、调试
  - 推荐配置：32GB 显存，用于 Fast-WAM released checkpoint 推理、Wan/Fast-WAM latent extraction
  - 更稳配置：40GB-80GB 显存，用于更长视频窗口、更大 batch、LoRA / adapter 微调
- 机械臂：单臂 UR5-CB3
- 控制库：`ur_rtde`，纯 Python，无 ROS
- 夹爪：Robotiq 2F-85
- 夹爪控制：Socket 直连 63352 端口，ASCII 协议 `SET` / `GET`
- 相机：RealSense D435 全局视角 + RealSense D415 腕部视角
- 相机读取：`pyrealsense2` 读取 RGB
- 遥操作：已有主臂实现对 UR5 的力控随动映射
- 数据格式：现有采集代码产出 LeRobot Dataset v3.0
- 动作约定：旋转统一使用 `xyzw` 顺序四元数，下发 UR5 前用 `scipy.spatial.transform.Rotation.from_quat(...).as_rotvec()` 转轴角；采集、训练、推理两侧必须保持一致
- 目标物体：豆腐、果冻、海绵、软糕点等超柔软块体类物体

## 已确认边界

- 第一篇论文可以不使用灵巧手，先用平行夹爪把主线做扎实。
- 第一篇论文可以使用预训练大模型 backbone，但不从零训练 Wan2.2-5B、通用 VLA 或通用视频 WAM。
- 当前系统具备真实夹持力反馈，因此可以把夹持力、压力垫信号和抓取结果联合建模。
- RGB-D 相机可作为物体定位、尺寸估计、姿态估计和实验记录工具，但不应喧宾夺主。
- 本地 RTX 4070 8GB 不承担 Fast-WAM/Wan 大模型训练；正式 visual latent extraction 优先放到云端 32GB 及以上 GPU。

## 两条递进式研究路线

本课题建议拆成两条有先后顺序的路线，而不是一开始同时做完所有场景。

### 路线一：易损物体抓取

这是第一阶段主线，也是最适合作为第一篇论文的方向。

目标是让机器人面对豆腐、果冻、海绵、软糕点等超柔软易损物体时，先通过轻触或轻夹获得早期视觉-触觉状态，再预测不同抓取参数的后果，选择成功率高且损伤风险低的动作。

核心任务：

- 建立易损物体抓取数据采集流程。
- 定义损伤指标，例如破裂、压痕、永久形变、高度恢复率、渗水或质量损失。
- 用压力垫、夹持力、夹爪位移和 RGB-D 物体状态预测滑移风险、损伤风险和形变恢复。
- 对多个候选抓取参数进行后果预测，并选择低风险参数。
- 在真实 UR5 + 平行夹爪平台上验证成功率提升和损伤率下降。

这一阶段的模型可以表述为：

```text
物体视觉状态 + 早期触觉状态 + 候选抓取参数
  -> 抓取成功 / 滑移风险 / 损伤风险 / 形变恢复
  -> 安全抓取参数
```

### 路线二：人机递交中的损伤/安全感知抓取

这是第二阶段扩展，建议建立在路线一的数据协议、触觉模型和安全抓取参数生成方法之上。

路线一只关心“如何不把物体抓坏并成功拿起来”。路线二进一步加入人：机器人不仅要保护物体，还要在人机递交时保护人的手，避免夹伤、碰撞、过大接触力和不稳定交接。

核心任务：

- 引入人手位置、接近方向、递交姿态和安全区域。
- 将损伤风险从“物体损伤”扩展为“物体损伤 + 人手安全 + 递交稳定性”。
- 利用 RGB-D 或人体/手部关键点检测估计人手状态。
- 利用夹持力和压力垫信号判断人是否已经接住、是否发生拉扯、是否需要释放。
- 设计递交策略：接近、等待、轻触确认、释放、异常中止。

这一阶段的模型可以表述为：

```text
物体状态 + 人手状态 + 早期触觉/力觉状态 + 候选递交动作
  -> 物体损伤风险 / 人手接触风险 / 交接成功率
  -> 安全递交动作
```

### 推荐先后顺序

建议先完成路线一，再做路线二。

原因是路线一能先把最核心、最可控的部分做出来：触觉采集、损伤标签、候选动作后果预测、安全抓取参数筛选。路线二只是把同一套思想扩展到更复杂的人机交互场景，新增的是人手状态、安全区域和交接时序。如果路线一没有稳定闭环，直接做人机递交会同时面对物体损伤、人身安全、动态交互和伦理风险，实验难度会明显上升。

更合理的论文/项目递进是：

```text
第一篇：面向超柔软易损物体的探索性触觉与损伤感知抓取
第二篇：面向人机递交的视触觉安全抓取与释放策略
```

## 第一阶段目标

先完成一个最小闭环：

```text
轻触采样
  -> 视觉-触觉状态编码
  -> 候选抓取参数后果预测
  -> slip / damage / deformation risk 估计
  -> 安全抓取参数生成或筛选
  -> 抬升与搬运验证
```

第一阶段采用 Fast-WAM-style 大模型路线：借用预训练视觉/视频/动作 backbone，训练本课题自己的 tactile-force adapter、candidate action encoder 和 outcome head。不追求灵巧手复杂操作，不从零训练 VLA 或通用 WAM。

当前短期目标进一步明确为：

```text
1. 先在 RTX 5090D 服务器上跑通官方 Fast-WAM 环境、released checkpoint 和基础推理/评估链路。
2. 理解 Fast-WAM/Wan latent backbone 的输入、输出和 checkpoint 组织方式。
3. 将官方 backbone 思路迁移为本项目的 frozen visual/video latent extractor。
4. 用 UR5 + GELLO + RGB-D + 触觉/力觉 + 易损标签训练 Fast-WAM-style latent outcome prediction。
```

也就是说，短期不是完整复现官方 Fast-WAM 大规模训练，而是先复现其可运行环境和 released checkpoint，再把可复用的 latent backbone 接入本课题数据流。

更具体地，第一篇论文建议聚焦：

- 轻触 / 轻夹过程中的触觉与夹持力信号采集
- 从早期触觉信号和 RGB-D 物体状态估计物体顺应性、滑移倾向和损伤风险
- 对多个候选抓取参数预测未来接触后果，并选择更安全的夹持力、闭合距离、接触位置或抬升策略
- 在真实 UR5 + 平行夹爪平台上验证抓取成功率和损伤降低效果

## WAM 融合定位

本项目中的 WAM 建议采用 Fast-WAM-style latent outcome prediction：

```text
视觉 / 视频 latent:
  来自冻结的 Fast-WAM / Wan / WALL / OpenPI backbone，或 RGB-D 特征编码器

触觉-力觉 latent:
  轻触阶段夹持力曲线、压力垫序列、夹爪开度和闭合速度

动作 latent:
  候选夹持力、闭合距离、闭合速度、抬升速度

后果预测:
  抓取成功率、滑移风险、损伤风险、形变量、恢复率
```

第一版模型先做 `visual_latent + tactile_force_latent + candidate_action_latent -> future_outcome`，即对候选动作打分。推理时不生成未来视频；未来视频建模只作为可选训练信号或预训练 backbone 的来源。

## 当前工程路线

当前项目采用 `FastWAM-Fragile` 工程骨架：

```text
frozen visual/video backbone
  -> visual_latent

task instruction
  -> text_instruction_latent

tactile-force adapter
  -> tactile_force_latent

candidate action encoder
  -> action_latent

DiT-style fusion transformer
  -> success / slip / damage / deformation / release safety
```

当前文本指令用于条件化候选动作后果预测和安全动作选择；它还不是完整的文本到动作生成器。真实动作仍需要来自 GELLO 示范、VLA proposal 或搜索采样的 candidate action chunks。

工程文件：

- [configs/fastwam_fragile.yaml](configs/fastwam_fragile.yaml)：训练与模型配置
- [datasets/episode_schema.md](datasets/episode_schema.md)：episode / JSONL 数据格式
- [docs/gello_to_fastwam_pipeline.md](docs/gello_to_fastwam_pipeline.md)：GELLO/LeRobot 数据到 FastWAM-Fragile 的转换流程
- [docs/fastwam_fragile_architecture.md](docs/fastwam_fragile_architecture.md)：架构说明
- [models/fastwam_fragile.py](models/fastwam_fragile.py)：Fast-WAM-style outcome model
- [scripts/extract_fastwam_visual_latents.py](scripts/extract_fastwam_visual_latents.py)：服务器端 Fast-WAM/Wan visual latent extraction
- [train.py](train.py)：训练入口

## 项目文档

- [research_brief.md](research_brief.md)：初始文献脉络、工作流程、网络选择、数据采集、具身智能/VLA 关系
- [docs/fastwam_short_term_plan.md](docs/fastwam_short_term_plan.md)：短期执行目标：先复现官方 Fast-WAM，再迁移到本项目
- [docs/fastwam_server_setup_5090.md](docs/fastwam_server_setup_5090.md)：RTX 5090D 服务器部署与数据集存放建议
- [docs/fastwam_latent_adapter.md](docs/fastwam_latent_adapter.md)：Fast-WAM visual latent adapter 的使用范围和运行命令
