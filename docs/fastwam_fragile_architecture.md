# FastWAM-Fragile 工程骨架

目标不是从零复现 Fast-WAM 或 Wan2.2-5B，而是把 Fast-WAM 的核心思想迁移到本课题：

```text
训练时显式学习 state-action -> future latent 的世界动态
推理时可在 latent 空间想象未来接触状态，而不是必须生成 RGB 未来视频
再从预测的 future contact latent 判断损伤、滑移和形变
```

## 模块划分

```text
RGB-D / video
  -> frozen Fast-WAM / Wan / VLA visual backbone
  -> visual_latent

pressure / force / gripper sequence
  -> tactile-force adapter
  -> tactile_latent

task instruction
  -> task parser / planner
  -> task_spec

candidate action chunk
  -> task-conditioned candidate action generator
  -> MCF-Proto action encoder
  -> action_latent

visual_latent + tactile_latent + action_latent
  -> FutureLatentDiffusion / FutureContactDiT
  -> predicted_future_contact_latent
  -> outcome heads
  -> success / slip / damage / deformation / release safety
```

文本指令不直接改变 WAM 对损伤、滑移和形变的物理预测。它用于解析任务目标、目标物体和操作阶段，生成 `task_spec`，再指导 candidate action generator 产生对应任务的候选动作。损伤安全是易损物体抓取的固定硬约束，不由 “抓豆腐” 或 “轻轻抓豆腐” 这类措辞改变。

## 为什么不复现 Wan2.2-5B

Wan2.2-5B video DiT backbone 是预训练视频生成/视频理解基座，参数量和训练数据规模都远超当前课题需要。本文路线只借用它的视觉时空表征能力，不从零训练。

本课题的新增贡献应该是：

- 易损物体精细数据集
- 触觉/力觉 adapter
- 候选动作后果预测
- 损伤/滑移/递交安全标签
- UR5 + 平行夹爪真实闭环验证

## FutureContactDiT 在本项目中的作用

DiT, Diffusion Transformer，原本是用 Transformer 做 diffusion denoising 的结构。本项目不从零训练通用视频生成模型，但需要显式学习动作条件下的未来接触 latent，因此新增：

```text
FutureLatentDiffusion / FutureContactDiT
```

它的输入和输出为：

```text
current visual_latent
probe tactile_force_latent
candidate action_latent
noise z_t
diffusion timestep t
  -> denoised future_contact_latent
  -> outcome heads
```

训练时，`future_contact_latent` 可以由抓取后窗口的视觉 latent、触觉/力觉 latent 或二者拼接得到。这样模型不是只做 `state + action -> label`，而是先学习 `state + action -> future latent`，再从未来 latent 判断 `damage / slip / deformation / success`，更符合 WAM 的世界动态建模定位。

## 第一版数据单元

每个 episode 建议保存：

```text
episode_id
task_instruction
object_type
object_id
rgb_path / depth_path / video_path
visual_latent_path
pressure_sequence_path
force_sequence_path
gripper_sequence_path
robot_state_sequence_path
candidate_action_path
executed_action_path
success
slip
damage
deformation
recovery_ratio
force_overshoot
release_ready
human_unsafe
```

## 推荐训练顺序

1. 先用预计算 `visual_latent` 跑通 `train.py`。
2. 再接真实 Fast-WAM/Wan/OpenPI/WALL 视觉 backbone。
3. 增加 future window 数据，构造 `future_contact_latent` 或 `future_visual_latent`。
4. 训练 `FutureContactDiT` 做 latent denoising + outcome prediction。
5. 数据稳定后，再考虑 LoRA 微调视觉/action backbone。
