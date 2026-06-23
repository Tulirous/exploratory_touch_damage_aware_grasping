# GELLO 到 FastWAM-Fragile 数据流程

本文档把 `GELLO 使用教程指南.txt` 中的遥操作采集流程，接到当前项目的 FastWAM-Fragile 训练流程。

核心链路：

```text
GELLO 遥操作
  -> LeRobot V3.0 episode
  -> 补充触觉/力觉/损伤标签
  -> 构建 FastWAM-Fragile JSONL manifest
  -> 提取 visual_latent.npy
  -> train.py 训练 outcome model
```

## 1. GELLO 当前采集能力

指南中的 `LeRobotSaveInterface` 当前保存：

```text
observation.state: shape (6,)
action: shape (6,)
observation.images.base: video, shape (3, 480, 640)
observation.images.wrist: video, shape (3, 480, 640)
fps: 30
```

含义：

- `base`：D435 全局相机，端口 5001。
- `wrist`：D415 腕部相机，端口 5000。
- `observation.state`：UR5 关节状态，当前示例是 6 维。
- `action`：GELLO 遥操作动作，当前示例是 6 维。

项目固定机器人栈：

```text
UR5-CB3
ur_rtde 纯 Python 控制，无 ROS
Robotiq 2F-85 夹爪，Socket 63352 ASCII SET/GET
D435 全局 RGB + D415 腕部 RGB，pyrealsense2 读取
动作旋转统一使用 xyzw 四元数
下发 UR5 前使用 scipy R.from_quat(...).as_rotvec() 转轴角
```

所有采集、训练和闭环推理代码必须保持同一套四元数顺序和轴角转换约定，避免左右手系或 `wxyz/xyzw` 混用。

这个格式可以用于 VLA / imitation learning，但本课题还需要额外补：

```text
pressure sequence
gripper force sequence
gripper opening / velocity
candidate actions
damage / slip / deformation labels
```

## 2. 建议改造 LeRobot 采集字段

如果夹爪和压力垫数据能同步读取，建议把 LeRobot feature 扩展成：

```python
features = {
    "observation.state": {"dtype": "float32", "shape": (7,)},
    "action": {"dtype": "float32", "shape": (7,)},
    "observation.images.base": {"dtype": "video", "shape": (3, 480, 640)},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640)},
    "observation.tactile": {"dtype": "float32", "shape": (C,)},
    "observation.force": {"dtype": "float32", "shape": (1,)},
}
```

其中：

```text
7 维 action = 6 维 UR5 动作 + 1 维夹爪动作
C = 压力垫通道数
```

如果短期内无法把触觉/力觉写入 LeRobot，可先作为 sidecar 文件保存：

```text
data/raw/<episode_id>/tactile.npy
data/raw/<episode_id>/force.npy
data/raw/<episode_id>/gripper.npy
```

## 3. 易损物体 episode 标准流程

每条 episode 建议包含完整任务：

```text
1. 机械臂移动到物体上方
2. 接近候选抓取点
3. 轻触 / 轻夹 0.5-2 秒
4. 正式闭合夹爪
5. 抬升 5-10 cm
6. 保持 1-2 秒观察滑移
7. 放回或结束
8. 人工/半自动标注 success、slip、damage、deformation
```

按键流程仍沿用 GELLO：

```text
按住 S：录制当前 episode
松开 S：保存当前 episode
按 Q：consolidate 数据集并退出
```

每条 episode 结束后补一份标签：

```json
{
  "episode_id": "tofu_0001_0001",
  "object_type": "tofu",
  "object_id": "tofu_brand_a_01",
  "success": 1,
  "slip": 0,
  "damage": 0,
  "force_overshoot": 0,
  "deformation": 0.12,
  "recovery_ratio": 0.93
}
```

## 4. LeRobot episode 到本项目 manifest

FastWAM-Fragile 的训练入口读 JSONL manifest：

```text
data/manifests/train.jsonl
data/manifests/val.jsonl
```

每一行指向该 episode 的视频、触觉、力觉、动作、标签和视觉 latent：

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

可以用：

```bash
python scripts/build_manifest_from_gello.py \
  --episodes data/metadata/gello_episodes.jsonl \
  --latent-dir data/latents \
  --train-out data/manifests/train.jsonl \
  --val-out data/manifests/val.jsonl
```

## 5. candidate action 怎么来

第一版可以从 GELLO 实际执行动作生成候选动作：

```text
positive candidate:
  GELLO 实际执行的安全动作

nearby candidates:
  对夹爪闭合量、闭合速度、抬升速度做小扰动

unsafe candidates:
  采集中真实导致滑移/损伤的动作
```

如果第一阶段只有实际执行动作，没有多个候选动作，也可以临时设：

```text
K = 1
candidate_actions = executed_action[None, :, :]
```

## 6. visual_latent.npy 怎么生成

`visual_latent.npy` 是从 `base` / `wrist` 视频中提取的视觉世界状态表示：

```text
D435 base video + D415 wrist video
  -> frozen Fast-WAM/Wan-style visual backbone
  -> visual_latent.npy
```

不要从零训练 Wan2.2-5B。Wan/Fast-WAM 的作用只是：

```text
视频/RGB-D -> visual latent
```

FastWAM-Fragile 自己负责：

```text
visual latent + tactile-force latent + action latent -> outcome
```

先测试 downstream 训练流程时，可以用 dummy latent：

```bash
python scripts/extract_visual_latents.py \
  --manifest data/manifests/train.jsonl \
  --backbone dummy
```

正式实验时把 `dummy` 替换为真实 backbone adapter。

结合官方 Fast-WAM 的 LIBERO eval 代码，第一版 latent extraction 建议：

```text
D435 frame + D415 frame
  -> 各自 center-crop resize 到 224x224
  -> 横向拼接成 224x448
  -> normalize 到 [-1, 1]
  -> frozen Fast-WAM VAE encode
  -> video_expert.pre_dit
  -> mean-pool video_pre["tokens"]
  -> 保存为 shape (3072,) 的 visual_latent.npy
```

本地 RTX 4070 8GB 负责采集和轻量闭环客户端；正式 Fast-WAM/Wan latent extraction 优先在云端 32GB 及以上 GPU 上离线完成，再把 `.npy` 回传到本项目训练数据目录。

## 7. 训练流程

准备好 manifest 和 latents 后：

```bash
python train.py --config configs/fastwam_fragile.yaml
```

训练输入：

```text
visual_latent
task_instruction
tactile / force / gripper sequence
candidate action chunk
```

训练输出：

```text
success
slip
damage
force_overshoot
deformation
recovery_ratio
release_ready
human_unsafe
```

## 8. 当前最小可行闭环

```text
GELLO 采 100 条豆腐/海绵 episode
  -> 人工标注 success/slip/damage
  -> 暂时 K=1，用实际执行动作作为 candidate action
  -> 用 frozen visual backbone 提取 visual_latent
  -> train.py 训练 outcome head
  -> 检查 damage/slip AUC
```

如果这个闭环跑通，再加入：

- 多候选动作生成
- 夹爪力/开度控制
- 人机递交 release_ready 标签
- VLA/WALL/OpenPI 候选动作生成
