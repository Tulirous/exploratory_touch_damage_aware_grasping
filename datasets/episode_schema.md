# Episode Schema

本项目第一版使用 JSONL manifest。每一行对应一个 episode 或 trial。

```json
{
  "episode_id": "tofu_0001_0001",
  "task_instruction": "gently grasp the tofu without damaging it",
  "object_type": "tofu",
  "object_id": "tofu_brand_a_01",
  "rgb_path": "data/raw/tofu_0001/rgb.mp4",
  "depth_path": "data/raw/tofu_0001/depth.npz",
  "base_video_path": "data/lerobot/videos/base/episode_000001.mp4",
  "wrist_video_path": "data/lerobot/videos/wrist/episode_000001.mp4",
  "visual_latent_path": "data/latents/tofu_0001_visual.npy",
  "tactile_path": "data/raw/tofu_0001/tactile.npy",
  "force_path": "data/raw/tofu_0001/force.npy",
  "gripper_path": "data/raw/tofu_0001/gripper.npy",
  "robot_state_path": "data/raw/tofu_0001/robot_state.npy",
  "candidate_action_path": "data/raw/tofu_0001/candidate_actions.npy",
  "executed_action_path": "data/raw/tofu_0001/executed_action.npy",
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

## 与 GELLO / LeRobot V3.0 的对应关系

`GELLO 使用教程指南.txt` 中当前 LeRobot 采集字段为：

- `observation.images.base`: D435 全局相机视频，`(3, 480, 640)`, 30 FPS
- `observation.images.wrist`: D415 腕部相机视频，`(3, 480, 640)`, 30 FPS
- `observation.state`: UR5 关节状态，示例为 6 维
- `action`: GELLO 遥操作动作，示例为 6 维

本项目额外需要：

- `tactile_path`: 压力垫时序
- `force_path`: 夹持力时序
- `gripper_path`: 夹爪开度/速度时序
- `candidate_action_path`: 候选动作
- `labels`: success / slip / damage / deformation 等后果标签

数组约定：

- `visual_latent`: `[visual_latent_dim]` 或 `[num_visual_tokens, visual_latent_dim]`
- `tactile`: `[T, C]`
- `force`: `[T, 1]`
- `gripper`: `[T, 2]`，建议包含开度和速度；如果暂时没有夹爪数据，可用零数组占位，但必须在实验记录中注明
- `candidate_actions`: `[K, H, A]`
- `executed_action`: `[H, A]`

当前 GELLO 示例的 `A=6`。加入夹爪动作后建议 `A=7`。
