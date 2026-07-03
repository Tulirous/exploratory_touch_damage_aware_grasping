# Contact-Gated Latent WAM 模型框架

本文档记录当前课题的目标模型框架。当前方向不再是简单的：

```text
state + action -> outcome label
```

而是明确采用 WAM 形式：

```text
current state + candidate action
  -> predicted future contact latent
  -> damage / slip / deformation / success
```

推荐方法名：

```text
Contact-Gated Latent World-Action Modeling
```

或作为方法简称：

```text
Contact-Gated Latent-Only WAM
```

核心原则：

- 文本指令用于解析任务目标和目标物体，不直接改变损伤/滑移/形变的物理预测。
- 易损物体抓取中的安全约束是默认硬约束，不由“抓豆腐”或“轻轻抓豆腐”这类措辞改变。
- WAM 主干必须显式学习 `state + action -> future latent`，避免退化成普通 action-conditioned classifier。
- 推理时不强制生成 RGB future video，但需要在 latent 空间预测未来接触/视觉状态。

## 总体 Pipeline

```text
Text instruction
  -> Task Parser / Planner
  -> task_spec

task_spec + visual state + robot state
  -> Task-Conditioned Candidate Action Generator
  -> candidate action chunks

D435 global RGB + D415 wrist RGB
  -> frozen Fast-WAM / Wan visual backbone
  -> current_visual_latent

pressure + force + gripper sequence
  -> Contact-Gated TactileForceAdapter
  -> probe_contact_latent

candidate action chunks
  -> MCF-Proto Action Encoder
  -> action_latent

current_visual_latent + probe_contact_latent + action_latent
  -> FutureLatentDiffusion / FutureContactDiT
  -> predicted_future_contact_latent

predicted_future_contact_latent
  -> outcome heads
  -> success / slip / damage / deformation / recovery_ratio

outcomes
  -> Safety-Aware Action Selector
  -> selected UR5 + Robotiq action
```

## 1. Text Task Module

目标：

```text
task_instruction -> task_spec
```

文本模块不直接进入 WAM 的 damage/slip/deformation fusion。它负责把自然语言任务转成结构化任务描述，例如：

```json
{
  "task_type": "fragile_grasp",
  "object": "tofu",
  "target": null,
  "phase": "grasp",
  "safety_mode": "fragile_default"
}
```

示例：

```text
抓豆腐
  -> task_type=fragile_grasp, object=tofu

把豆腐放到盘子里
  -> task_type=pick_place, object=tofu, target=plate

把豆腐递给人
  -> task_type=handover, object=tofu, target=human
```

注意：

```text
"抓豆腐" 和 "轻轻抓豆腐" 不应该导致不同损伤容忍度。
```

对易损物体，低损伤、低滑移和力安全是默认约束。

推荐实现路线：

```text
第一版:
  rule-based parser + task/object vocabulary

增强版:
  sentence embedding prototype classifier
  例如 BAAI/bge-small-en-v1.5, intfloat/e5-small-v2, BAAI/bge-m3

研究版:
  LLM / VLM 输出 JSON task_spec，并经过严格 schema 校验
```

当前代码里的 `models/text_instruction_encoder.py` 是早期 text latent 方案。按当前设计，后续应将其替换或改造为 `TaskTextParser / LanguageTaskPlanner`，输出 `task_spec`，而不是直接给 `OutcomeTransformer` 提供 `text_latent`。

## 2. Task-Conditioned Candidate Action Generator

目标：

```text
task_spec + visual state + robot state
  -> candidate action chunks
```

这个模块回答：

```text
当前任务应该尝试哪些可执行动作？
```

它不是 WAM 本体，但它是闭环系统从文本到动作的必要前级。

候选动作来源：

```text
1. Motion primitive sampler
   approach / light touch / close / lift / hold

2. GELLO demonstration prior
   从人示范轨迹附近采样候选动作

3. VLA proposal
   由 VLA 根据文本和视觉提出动作，再交给 WAM 做安全过滤
```

第一阶段建议只支持易损物体抓取任务：

```text
fragile_grasp
pick_place_fragile
hold_without_slip
```

暂不支持：

```text
fold cloth
open drawer
general household manipulation
```

候选动作格式：

```text
candidate_actions: [B, K, H, A]
```

其中：

- `K`: 候选动作数量
- `H`: action horizon
- `A`: action dimension，当前 UR5 为 6，加入 Robotiq gripper action 后为 7

## 3. Frozen Visual / Video Backbone

目标：

```text
D435 base video + D415 wrist video
  -> frozen Fast-WAM / Wan-style backbone
  -> current_visual_latent
```

当前采用 Fast-WAM-style visual latent extraction：

```text
D435 frame + D415 frame
  -> 各自 center-crop resize 到 224x224
  -> 横向拼接成 224x448
  -> normalize 到 [-1, 1]
  -> Fast-WAM/Wan VAE image encoder
  -> video_expert.pre_dit
  -> pooled video_pre["tokens"]
  -> visual_latent.npy, shape (3072,)
```

这个模块提供当前视觉世界状态，但不直接生成 UR5 动作。

当前工程状态：

- `configs/fastwam_fragile.yaml` 默认 `visual_latent_dim: 3072`
- `scripts/extract_visual_latents.py --backbone dummy` 只用于测试
- `scripts/extract_fastwam_visual_latents.py` 用于服务器端调用官方 Fast-WAM 提取真实 latent
- backbone 默认冻结

## 4. Contact-Gated TactileForceAdapter

目标：

```text
pressure / force / gripper sequence
  -> probe_contact_latent
```

它编码轻触/轻夹阶段的接触诊断信号，是本课题的核心模块之一。

建议输入：

```text
tactile.npy: [T, C]
force.npy: [T, 1]
gripper.npy: [T, 2]
```

建议从普通 Transformer 序列编码升级为 contact-gated 表示：

```text
sequence branch:
  temporal conv / transformer

physics feature branch:
  contact onset time
  max force
  force slope
  pressure peak
  pressure concentration / pressure area
  gripper displacement at contact
  stiffness proxy = delta_force / delta_gripper

contact gate:
  用接触强度和力增长模式调制 future latent prediction
```

输出：

```text
probe_contact_latent
contact_gate
contact_physics_features
```

其中 `contact_gate` 用于调制 `FutureContactDiT`：

```text
contact_gate controls how strongly tactile/force evidence affects future latent denoising.
```

## 5. MCF-Proto Action Encoder

目标：

```text
candidate action chunks
  -> motion-centric action_latent
```

当前 `models/action_encoder.py` 是普通 action chunk encoder。后续应升级为：

```text
Motion-Centric Candidate Action Encoding, MCF-Proto
```

它不应只对原始动作做 mean pooling，而应提取局部运动 primitive：

```text
approach direction
contact offset
closing displacement
closing velocity
lift direction
lift velocity
relative rotation
gripper width change
motion smoothness
```

动作约定：

```text
旋转统一使用 xyzw 四元数
下发 UR5 前用 scipy R.from_quat(...).as_rotvec() 转轴角
```

MCF-Proto 的作用：

```text
把 GELLO / VLA / motion sampler 产生的不同动作来源统一到局部运动表示，
让 WAM 学习“这个局部运动会导致什么未来接触状态”。
```

## 6. FutureLatentDiffusion / FutureContactDiT

目标：

```text
current_visual_latent + probe_contact_latent + action_latent
  -> predicted_future_contact_latent
```

这是区别于普通 outcome classifier 的关键 WAM 模块。

输入：

```text
current_visual_latent
probe_contact_latent
contact_gate
action_latent
noisy_future_latent z_t
diffusion timestep t
```

输出：

```text
denoised future_contact_latent
```

训练目标：

```text
diffusion denoising loss:
  predict noise epsilon
  or predict clean future latent x0

outcome auxiliary loss:
  future_contact_latent -> success / slip / damage / deformation
```

future latent 可以来自：

```text
1. future tactile-force latent
   抓取后窗口的 force / pressure / gripper 序列编码结果

2. future visual latent
   抓取后 D435/D415 视频窗口的 Fast-WAM/Wan latent

3. future contact-state latent
   concat(future_tactile_latent, future_visual_latent)
```

第一版建议：

```text
future_contact_latent = future tactile-force latent
```

原因：

- 损伤和滑移更直接体现在未来接触/力觉状态
- 比 RGB future video 更轻
- 更符合 latent-only WAM

第二版再加入：

```text
future_visual_latent
```

用于增强形变和接触区域变化建模。

## 7. Outcome Heads

目标：

```text
predicted_future_contact_latent
  -> physical outcomes
```

输出：

```text
binary:
  success
  slip
  damage
  force_overshoot
  release_ready
  human_unsafe

regression:
  deformation
  recovery_ratio
```

第一阶段重点：

```text
success
slip
damage
force_overshoot
deformation
recovery_ratio
```

第二阶段人机递交再启用：

```text
release_ready
human_unsafe
```

## 8. Safety-Aware Action Selector

目标：

```text
predicted outcomes for K candidate actions
  -> selected action
```

选择函数应是固定安全目标，不由文本措辞自由改变：

```text
score_k =
  success_prob_k
  - lambda_damage * damage_prob_k
  - lambda_slip * slip_prob_k
  - lambda_deform * deformation_k
  - lambda_force * force_overshoot_prob_k
```

对易损物体：

```text
damage / force_overshoot 是硬约束或高权重惩罚项
```

动作执行：

```text
selected candidate action
  -> ur_rtde
  -> UR5-CB3
  -> Robotiq 2F-85 socket control
```

## 9. 训练数据要求

每条 episode 需要至少包含：

```text
episode_id
task_instruction
task_spec
object_type
object_id
base_video_path
wrist_video_path
visual_latent_path
future_visual_latent_path optional
tactile_path
force_path
gripper_path
future_tactile_path
future_force_path
future_gripper_path
candidate_action_path
executed_action_path
labels
```

第一版数组约定：

```text
current_visual_latent.npy:       [3072]
future_visual_latent.npy:        [3072] optional
probe_tactile/force/gripper:     [T_probe, C]
future_tactile/force/gripper:    [T_future, C]
candidate_actions.npy:           [K, H, A]
executed_action.npy:             [H, A]
```

如果第一阶段只有实际执行动作：

```text
K = 1
candidate_actions = executed_action[None, :, :]
```

但为了体现 WAM 的 action-conditioned future modeling，后续应逐步增加：

```text
nearby action perturbations
unsafe historical actions
motion primitive candidates
```

## 10. 训练目标

总损失建议：

```text
L =
  L_diffusion_future_latent
  + lambda_binary * L_binary_outcomes
  + lambda_regression * L_regression_outcomes
  + lambda_rank * L_candidate_ranking optional
```

其中：

```text
L_diffusion_future_latent:
  predicted future latent vs target future latent

L_binary_outcomes:
  success / slip / damage / force_overshoot BCE

L_regression_outcomes:
  deformation / recovery_ratio MSE or SmoothL1

L_candidate_ranking:
  encourage safer executed/successful actions to score above damaging/slipping actions
```

## 11. 与 Fast-WAM / LaWAM / RepWAM 的关系

本项目不从零训练通用视频 WAM，也不复现 Wan2.2-5B。

借鉴点：

```text
Fast-WAM:
  利用预训练视频/动作世界模型思想，但推理不依赖显式 RGB future video generation

LaWAM:
  强调 latent future/subgoal，避免像素级未来生成的高延迟

RepWAM:
  强调 representation-centric future/action token modeling
```

本项目落点：

```text
Contact-gated latent future prediction for fragile object grasping.
```

核心贡献不在通用模型规模，而在：

```text
易损物体数据
探索性轻触
触觉/力觉 contact gate
motion-centric candidate action encoding
future contact latent diffusion
真实 UR5 + Robotiq 闭环验证
```

## 12. 当前工程状态

当前代码仍处于过渡状态：

```text
已实现:
  FastWAMFragile wrapper
  TactileForceAdapter
  ActionChunkEncoder
  OutcomeTransformer
  TextInstructionEncoder early version
  Fast-WAM visual latent adapter

需要重构:
  TextInstructionEncoder -> TaskTextParser / LanguageTaskPlanner
  ActionChunkEncoder -> MCF-Proto Action Encoder
  OutcomeTransformer -> FutureContactDiT + OutcomeHeads
  Dataset -> 增加 future window / future_contact_latent target
  Train loop -> 增加 diffusion denoising loss
```

因此，当前实现可以作为 pipeline smoke test，但最终论文模型应按本文档的 Contact-Gated Latent WAM 架构推进。
