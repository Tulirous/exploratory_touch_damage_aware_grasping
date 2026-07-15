# SMPL-X/MANO Hand Sequence Schema

Stage 1 不直接读取 RGB 视频，而是读取离线提取并完成时序跟踪的单手序列。每个 `NPZ` 对应一条连续手部 track。

## NPZ 数组

```text
hand_state: [T, 24], float32, required for HOT3D pilot
joints_3d:  [T, 21, 3], float32, recommended
contact:    [T, 5], float32/bool, optional
valid:      [T], bool, recommended for manifest filtering
```

HOT3D pilot 默认 `hand_state` 布局：

```text
0:3    wrist_translation，建议以物体坐标系表达，单位为米
3:9    wrist_rotation_6d
9:24   HOT3D 提供的 15 个 MANO PCA pose coefficients
```

HOT3D 的 `mano_pose.thetas` 是 15 维 PCA 系数，不是 15 个关节的轴角。
转换脚本默认将腕部位姿表达在每个 hand track 第一帧的腕部坐标系中；
正式物体交互实验再切换到 object-relative frame。

`shape/betas` 描述人物手型，在同一 track 中应保持不变，第一版不作为未来动态预测目标。左、右手建议分别形成 track；左手可以镜像到统一右手规范，但必须在 manifest 中保留 `handedness`。

## Window Manifest

JSONL 每行对应一个训练窗口：

```json
{
  "episode_id": "ego4d_clip_000123",
  "sequence_path": "data/smplx_hand/ego4d_clip_000123_right.npz",
  "handedness": "right",
  "start_index": 120,
  "object_id": "tofu_01",
  "task_instruction": "gently pick up the tofu"
}
```

按照默认配置，`start_index` 后 4 帧为 `hand_context`，再后 12 帧为 `hand_future`。生成 manifest 时必须过滤严重遮挡、追踪切换、缺少尺度或无效姿态窗口。

## 边界说明

RGB 到 SMPL-X/MANO 的恢复工具尚未在本子项目中绑定。实际实验应选定并记录 HaMeR、Dyn-HaMR 或其他后端、checkpoint、相机标定与平滑参数；不能把自动恢复结果当作无噪声真值。

HOT3D-Clips 是例外：pilot 直接读取其逐帧 GT MANO-PCA 标注，不经过 RGB
手部恢复模型。获得合规 MANO assets 后，可以由 `mano_beta`、`mano_pca` 和
`wrist_xform` 前向生成 21 个关节点，用于几何指标。
