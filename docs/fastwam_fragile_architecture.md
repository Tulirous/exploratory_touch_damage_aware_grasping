# FastWAM-Fragile 工程骨架

目标不是从零复现 Fast-WAM 或 Wan2.2-5B，而是把 Fast-WAM 的核心思想迁移到本课题：

```text
训练时利用未来后果监督学习世界表征
推理时不生成未来视频
直接预测候选动作的损伤、滑移和递交安全后果
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
  -> text instruction encoder
  -> text_latent

candidate action chunk
  -> action encoder
  -> action_latent

visual_latent + text_latent + tactile_latent + action_latent
  -> DiT-style fusion transformer
  -> outcome heads
  -> success / slip / damage / deformation / release safety
```

文本指令在第一版中作为 outcome prediction 的条件输入，用于区分任务目标和风险偏好，例如 “gently grasp without damaging” 与普通抓取。动作本身仍由 GELLO / VLA / 搜索器生成候选，再由 outcome model 选择风险更低的候选动作。

## 为什么不复现 Wan2.2-5B

Wan2.2-5B video DiT backbone 是预训练视频生成/视频理解基座，参数量和训练数据规模都远超当前课题需要。本文路线只借用它的视觉时空表征能力，不从零训练。

本课题的新增贡献应该是：

- 易损物体精细数据集
- 触觉/力觉 adapter
- 候选动作后果预测
- 损伤/滑移/递交安全标签
- UR5 + 平行夹爪真实闭环验证

## DiT 在本项目中的作用

DiT, Diffusion Transformer，原本是用 Transformer 做 diffusion denoising 的结构。在本项目中，不需要完整 diffusion 采样；可以借鉴 DiT 的 token 化和条件融合方式：

```text
visual token
tactile token
action token
task token
  -> transformer fusion
  -> future outcome
```

所以这里的独有 DiT 更准确叫：

```text
Action-Conditioned Visuo-Tactile Outcome Transformer
```

它不是生成未来视频，而是预测候选动作会带来的后果。

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
3. 冻结 backbone，只训练 tactile adapter、action encoder、fusion transformer 和 outcome heads。
4. 数据稳定后，再考虑 LoRA 微调视觉/action backbone。
