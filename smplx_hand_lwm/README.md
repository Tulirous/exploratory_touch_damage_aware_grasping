# SMPL-X Hand Latent World Model

这个独立子项目实现以下 Stage 1 研究假设：

```text
将 LaWAM 的视觉 latent feature F
替换为 SMPL-X/MANO hand state H

(H_context, H_future)
  -> Hand-IDM posterior
  -> latent hand action z_h
  -> Hand World Model decoder
  -> predicted H_future
```

它不会修改当前 `latent_action_idm` 或 `FastWAMFragile` 训练路径。当前平行夹爪实验继续使用原路线；本目录面向后续 Linker Hand O6 扩展。

## 目录

```text
smplx_hand_lwm/
  configs/stage1_hand_lwm.yaml
  datasets/hand_sequence_dataset.py
  datasets/schema.md
  docs/model_architecture.md
  docs/experiment_plan.md
  models/inverse_dynamics.py
  models/hand_world_model.py
  models/stage1_hand_lwm.py
  models/losses.py
  scripts/smoke_test_stage1.py
  train_stage1.py
```

## 当前已实现

- structured hand sequence Transformer IDM；
- Gaussian latent hand action；
- Transformer future trajectory decoder；
- hand state、3D joints、contact auxiliary heads；
- temporal velocity/acceleration、KL 等损失；
- JSONL + NPZ 数据接口；
- 最小前向与反向 smoke test；
- Stage 1 训练入口。

## 尚未接入

- RGB/ego video 到 SMPL-X/MANO 的真实提取后端；
- 合规 MANO assets 与可微 forward kinematics；
- Linker Hand O6 重定向；
- 真实机器人闭环验证。

因此当前代码只证明模型与数据接口可运行，不能声称已经完成 ego 数据训练或机器人验证。

## 快速检查

```bash
python -m smplx_hand_lwm.scripts.smoke_test_stage1
```

## HOT3D-Clips pilot

HOT3D Quest 3 tar 下载完成后，先转换 3 个 clips 做小规模测试：

```bash
python -m smplx_hand_lwm.scripts.prepare_hot3d \
  --clips-dir /data/chy/hot3d_clips/train_quest3 \
  --output-dir /data/chy/hot3d_hand_lwm/tracks \
  --train-manifest /data/chy/hot3d_hand_lwm/hot3d_hand_train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/hot3d_hand_val.jsonl \
  --handedness right \
  --max-clips 3
```

默认状态是 24 维：3D 腕部平移、6D 腕部旋转和 HOT3D 的 15 维 MANO
PCA pose。pilot 默认只用右手，避免模型尚未加入 handedness token 时混合左右手；
指定 `--handedness both` 会生成独立左右手 tracks。训练/验证按 clip 划分。

直接用服务器上的 manifest 启动 pilot 训练：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hand_lwm.yaml \
  --train-manifest /data/chy/hot3d_hand_lwm/hot3d_hand_train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/hot3d_hand_val.jsonl \
  --batch-size 32 \
  --num-workers 4 \
  --epochs 100
```

若服务器出现 DataLoader shared-memory/worker 错误，将 `--num-workers` 改为
`0`；这只影响数据加载速度，不改变模型。

训练完成后评估 posterior reconstruction、Last Pose、Constant Velocity 和
shuffle-z：

```bash
python -m smplx_hand_lwm.evaluate_stage1 \
  --checkpoint checkpoints/smplx_hand_lwm/stage1/best.pt \
  --manifest /data/chy/hot3d_hand_lwm/hot3d_hand_val.jsonl \
  --output /data/chy/hot3d_hand_lwm/pilot_metrics.json
```

这里的 posterior 使用了真实未来状态来生成 latent action，因此只验证 Stage 1
重建和 latent usage；它不是只输入当前状态的未来预测结果。

## Pilot50 与 Stage 2 latent prior

先转换 50 个 clips：

```bash
python -m smplx_hand_lwm.scripts.prepare_hot3d \
  --clips-dir /data/chy/hot3d_clips/train_quest3 \
  --output-dir /data/chy/hot3d_hand_lwm/pilot50/tracks \
  --train-manifest /data/chy/hot3d_hand_lwm/pilot50/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --handedness right \
  --max-clips 50
```

重新训练加强腕部轨迹的 Stage 1。该配置只对腕部 translation 使用
constant-velocity anchor，并将 wrist translation loss 权重提高到 5：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_pilot50.yaml
```

Stage 1 通过后，训练只看当前 4 帧的 latent-action prior：

```bash
python -m smplx_hand_lwm.train_stage2_prior \
  --config smplx_hand_lwm/configs/stage2_hand_prior.yaml
```

评估真正的 current-only rollout：

```bash
python -m smplx_hand_lwm.evaluate_stage2_prior \
  --prior-checkpoint /data/chy/hot3d_hand_lwm/pilot50/stage2_prior_checkpoints/best.pt \
  --stage1-checkpoint /data/chy/hot3d_hand_lwm/pilot50/stage1_checkpoints/best.pt \
  --manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --output /data/chy/hot3d_hand_lwm/pilot50/stage2_val_metrics.json
```

`current_only_prior/*` 才是没有读取未来的预测结果；
`posterior_teacher/*` 读取真实未来，只作为 Stage 1 重建上界。

数据格式见 `datasets/schema.md`，模型设计和实验计划分别见 `docs/model_architecture.md` 与 `docs/experiment_plan.md`。
