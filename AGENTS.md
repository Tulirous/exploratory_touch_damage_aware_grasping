# AGENTS.md

本文件用于指导后续 coding / research agent 在本项目中工作。项目当前方向已经从“小模型过渡”调整为 **Fast-WAM-style 大模型路线**，但不从零训练通用视频模型或通用 VLA。

## 项目目标

课题名称：

```text
Exploratory Touch for Damage-Aware Grasp Synthesis of Ultra-Soft Fragile Objects
面向超柔软易损物体的探索性触觉与损伤感知抓取参数生成
```

当前核心路线：

```text
预训练 Fast-WAM / Wan / WALL / OpenPI backbone
  -> visual/video latent

GELLO + UR5 + 平行夹爪 + RGB-D + 压力/力觉
  -> tactile-force latent

candidate action chunk
  -> action latent

visual latent + tactile-force latent + action latent
  -> damage / slip / deformation / handover safety outcome
```

第一阶段只做 **易损物体抓取**。第二阶段再扩展到 **人机递交中的损伤/安全感知抓取**。

## 当前硬件与数据条件

- 机械臂：UR5
- 当前执行器：平行夹爪
- 后续执行器：Linker Hand O6，仍作为后续扩展
- 遥操：GELLO
- 相机：D415 腕部相机 + D435 全局相机
- 数据格式：LeRobot V3.0 episode + 本项目 sidecar 文件
- 算力：A100
- 目标物体：豆腐、果冻、海绵、软糕点等超柔软易损物体

GELLO 指南位于：

```text
GELLO 使用教程指南.txt
```

## 当前工程结构

重要文件：

```text
README.md
research_brief.md
GELLO 使用教程指南.txt
configs/fastwam_fragile.yaml
docs/fastwam_fragile_architecture.md
docs/gello_to_fastwam_pipeline.md
datasets/episode_schema.md
datasets/fragile_episode_dataset.py
models/fastwam_fragile.py
models/tactile_force_adapter.py
models/action_encoder.py
models/outcome_transformer.py
scripts/build_manifest_from_gello.py
scripts/extract_visual_latents.py
train.py
```

`train.py` 当前训练的是 FastWAM-Fragile outcome model，不是完整 Fast-WAM 官方模型。

## 研究路线约束

必须遵守：

- 不从零训练 Wan2.2-5B。
- 不从零训练通用 VLA。
- 不把“生成未来视频”作为推理时必要步骤。
- 不把 WALL-OSS / OpenPI / Fast-WAM 误写成已经能直接判断豆腐损伤的现成模型。
- 论文核心贡献应放在：易损物体数据、触觉/力觉 adapter、候选动作后果预测、真实 UR5 闭环验证。

推荐表述：

```text
Fast-WAM-style latent outcome prediction
```

不推荐表述：

```text
从零训练大规模 WAM
```

## 数据流程

当前推荐流程：

```text
GELLO 遥操作
  -> LeRobot V3.0 episode
  -> 补充 pressure / force / gripper sidecar
  -> 标注 success / slip / damage / deformation
  -> scripts/build_manifest_from_gello.py
  -> data/manifests/train.jsonl, val.jsonl
  -> scripts/extract_visual_latents.py
  -> data/latents/<episode_id>_visual.npy
  -> train.py
```

当前 GELLO 指南中的 LeRobot 字段：

```text
observation.images.base: D435 全局相机，(3, 480, 640), 30 FPS
observation.images.wrist: D415 腕部相机，(3, 480, 640), 30 FPS
observation.state: 6 维 UR5 状态
action: 6 维 GELLO/UR5 动作
```

当前 `configs/fastwam_fragile.yaml` 默认 `action_dim: 6`。加入夹爪动作后，再改为 `action_dim: 7`。

## Manifest 规范

训练入口读取 JSONL manifest。每行必须至少包含：

```json
{
  "episode_id": "tofu_0001_0001",
  "task_instruction": "gently grasp the tofu without damaging it",
  "object_type": "tofu",
  "object_id": "tofu_brand_a_01",
  "base_video_path": "data/lerobot/videos/base/episode_000001.mp4",
  "wrist_video_path": "data/lerobot/videos/wrist/episode_000001.mp4",
  "visual_latent_path": "data/latents/tofu_0001_0001_visual.npy",
  "tactile_path": "data/raw/tofu_0001_0001/tactile.npy",
  "force_path": "data/raw/tofu_0001_0001/force.npy",
  "gripper_path": "data/raw/tofu_0001_0001/gripper.npy",
  "candidate_action_path": "data/raw/tofu_0001_0001/candidate_actions.npy",
  "executed_action_path": "data/raw/tofu_0001_0001/executed_action.npy",
  "labels": {
    "success": 1,
    "slip": 0,
    "damage": 0,
    "force_overshoot": 0,
    "release_ready": 0,
    "human_unsafe": 0,
    "deformation": 0.12,
    "recovery_ratio": 0.93
  }
}
```

数组约定见：

```text
datasets/episode_schema.md
```

## Visual Latent 策略

`visual_latent.npy` 是视觉/视频 backbone 输出的 latent。

当前脚本：

```bash
python scripts/extract_visual_latents.py --manifest data/manifests/train.jsonl --backbone dummy
```

`dummy` 只用于测试数据链路，不能用于正式实验。

正式路线：

```text
base_video + wrist_video
  -> frozen Fast-WAM / Wan / WALL / OpenPI backbone
  -> visual_latent.npy
```

后续实现 backbone adapter 时，应优先保证：

- 输入路径来自 manifest。
- 输出 shape 与 `configs/fastwam_fragile.yaml` 的 `visual_latent_dim` 一致。
- 大模型权重默认冻结。
- 如果做 LoRA / adapter，需要单独配置，不要改默认训练路径。

## 模型结构

当前模型：

```text
FastWAMFragile
  TactileForceAdapter
  ActionChunkEncoder
  OutcomeTransformer
```

这里的 `OutcomeTransformer` 是 DiT-style fusion transformer，但不是完整 diffusion video generator。它的作用是融合：

```text
visual token
tactile-force token
candidate action token
```

输出：

```text
success
slip
damage
force_overshoot
release_ready
human_unsafe
deformation
recovery_ratio
```

## 训练命令

在 manifest 和 visual latents 准备好后：

```bash
python train.py --config configs/fastwam_fragile.yaml
```

修改模型或数据代码后，至少运行：

```bash
python3 -m py_compile train.py scripts/build_manifest_from_gello.py scripts/extract_visual_latents.py datasets/fragile_episode_dataset.py models/tactile_force_adapter.py models/action_encoder.py models/outcome_transformer.py models/fastwam_fragile.py
```

如果没有真实数据，不要声称模型已验证；最多只能说明 pipeline 语法或 dummy latent 跑通。

## 论文与实验注意事项

第一阶段实验重点：

- damage prediction AUC / F1
- slip prediction AUC / F1
- deformation regression error
- grasp success rate
- real damage rate
- unseen object generalization

Baseline 建议：

- 固定抓取参数
- 无轻触直接抓取
- 原始压力阈值控制
- VLA 直接动作
- VLA/WAM + tactile-force outcome prediction

第二阶段再加入：

- release_ready
- human_unsafe
- premature release rate
- handover success rate

## 文献与术语约束

重要术语：

- VLA: Vision-Language-Action，负责动作生成
- WAM: World-Action Model，负责建模动作后果或未来状态
- DiT: Diffusion Transformer，不等于视频 tokenizer
- Fast-WAM: 训练时利用视频共训练，推理时不生成未来视频
- Wan2.2-5B: 预训练 video DiT backbone，不是本课题要复现的目标

如果用户问最新论文、模型开源状态或项目可部署性，必须联网核对最新仓库/模型页。

## 协作规则

- 修改文件前先读相关文件。
- 使用 `rg` 搜索。
- 手动编辑使用 `apply_patch`。
- 不要删除用户已有文件或撤销用户更改。
- 新增工程文件要保持与当前 `configs/fastwam_fragile.yaml` 一致。
- 不要把占位脚本描述成已完成真实 backbone 接入。

