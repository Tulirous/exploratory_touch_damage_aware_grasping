# LaWAM-Aligned Latent Action IDM

这个子项目用于复现 LaWAM Stage 1 的核心思想，并作为后续加入触觉/力觉模块的起点。当前目标不是直接训练 VLA，也不是生成未来 RGB 视频，而是学习：

```text
D435/D415 当前视觉 latent + D435/D415 未来视觉 latent
  -> latent_action
  -> 未来视觉 latent
```

`robot_state` 在当前 visual-only teacher 中不进入 IDM，也不参与 teacher 训练目标。它只保留为诊断字段，避免 droid_100 的机器人状态空间污染后续要迁移到 UR5 的 latent action。

## 与 LaWAM 的对齐点

LaWAM Stage 1 使用：

```text
frozen distilled DINOv3 ViT-B/16 encoder
current observation o_t -> DINO feature u_t
horizon observation o_T -> DINO feature u_T
IDM q(z | u_t, u_T) -> latent action z
LaWM decoder p(u_T | u_t, z) -> predicted future latent subgoal
```

本子项目当前默认也采用 DINOv3 ViT-B/16：

```yaml
dino:
  backend: transformers
  model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
  feature_mode: patch_tokens
```

如果服务器使用的是官方本地 checkpoint，或者 Hugging Face 上的实际 model id 有差异，只需要改 `model_name`。

## 当前版本

当前代码已经放弃 mean-pooling + MLP IDM，改为 patch-token Transformer 版：

```text
DINOv3 spatial patch tokens
  -> Transformer IDM posterior q(z | u_t, u_T)
  -> latent_action z
  -> Transformer LaWM decoder p(u_T | u_t, z)
  -> predicted future DINO patch tokens
```

工程上已经拆成可单独调试的模块：

```text
latent_action_idm/models/common.py
  VisualTokenProjector
  AdaLNBlock
  MLP

latent_action_idm/models/inverse_dynamics.py
  InverseDynamicsTransformer
  输入: visual_t, visual_future
  输出: latent_mu, latent_logvar, latent_action

latent_action_idm/models/latent_world_model.py
  LatentWorldModelDecoder
  输入: visual_t, latent_action
  输出: predicted_visual_future

latent_action_idm/models/stage1_lawam.py
  Stage1LaWAM
  组合: IDM + LaWM decoder + auxiliary state predictor
```

为了适配双相机，当前默认把两个视角的 tokens 沿 token 维拼接：

```text
base image:  200 tokens
wrist image: 200 tokens
two views:   400 tokens
token dim:   768
```

其中单视角 `200 tokens` 来自当前本地 DINOv3 ViT-B/16 checkpoint 的实际输出，通常可理解为 `196 patch tokens + 4 extra/register tokens`。因此后续代码不能硬编码 `392` 或 `196`。

v4 teacher 还加入了 camera-aware view embedding：

```text
前 200 tokens: base/global view embedding
后 200 tokens: wrist view embedding
```

训练配置默认使用 8 层 Transformer encoder 和 10 层 decoder，方便先在 droid_100 上稳定验证。正式复现 LaWAM 规模时，可以再把配置中的层数调高：

```yaml
model:
  encoder_layers: 24
  decoder_layers: 24
```

## 为什么先冻结 DINOv3

你的数据目前是 60 条、约 11262 帧。这个规模不适合训练或全量微调 DINOv3。正确做法是：

```text
DINOv3 frozen
train IDM / LaWM / tactile adapter
```

后续如果要增强领域适配，可以考虑：

```text
frozen DINOv3 + projection adapter
frozen DINOv3 + LoRA
patch-token ViewFusion Transformer
```

## Episode Manifest

先准备 episode-level JSONL，每行一条 LeRobot episode。对于当前 LeRobot v3.0 目录，可以自动生成：

```bash
python -m latent_action_idm.scripts.build_lerobot_episode_manifest \
  --lerobot-root latent_action_idm/datasets/ur5_handover_60ep \
  --out data/manifests/ur5_handover_60ep_episodes.jsonl \
  --state-dir data/processed/ur5_handover_60ep_states
```

生成后的每行包含：

```json
{
  "episode_id": "episode_000000",
  "base_video_path": ".../observation.images.base/chunk-000/file-000.mp4",
  "wrist_video_path": ".../observation.images.wrist/chunk-000/file-000.mp4",
  "base_frame_offset": 0,
  "wrist_frame_offset": 0,
  "robot_state_path": "data/processed/ur5_handover_60ep_states/episode_000000_state.npy"
}
```

`robot_state_path` 是 `.npy`，shape 为 `[T, 7]`，包含 6 维 UR5 关节状态和 1 维 `gripper_width`。

## Step 1: 提取 DINOv3 Latents

```bash
python -m latent_action_idm.scripts.extract_dino_latents \
  --config latent_action_idm/configs/dino_idm.yaml \
  --episode-manifest data/manifests/ur5_handover_60ep_episodes.jsonl \
  --output-dir data/latents/dino_idm \
  --train-out data/manifests/idm_train.jsonl \
  --val-out data/manifests/idm_val.jsonl \
  --overwrite
```

默认窗口：

```text
t_index -> future_index = t_index + 30
stride = 15
```

30 FPS 下相当于用当前帧预测 1 秒后的状态，每 0.5 秒采一个样本。你的 60 条数据当前约生成 655 个训练窗口。

## Step 2: 训练 Stage-1 LaWAM

```bash
python -m latent_action_idm.train_idm \
  --config latent_action_idm/configs/dino_idm.yaml
```

输出：

```text
checkpoints/latent_action_idm/latest.pt
checkpoints/latent_action_idm/best.pt
```

如需从重构前的 checkpoint 继续训练：

```bash
python -m latent_action_idm.train_idm \
  --config latent_action_idm/configs/dino_idm.yaml \
  --resume checkpoints/latent_action_idm/best.pt
```

## 模块接口

训练样本包含：

```text
visual_t: [400, 768]
visual_future: [400, 768]
state_t: [7]         # diagnostic only for visual-only teacher
state_future: [7]    # diagnostic only for visual-only teacher
```

模型输出：

```text
latent_action: [128]
predicted_state_future: [7]      # diagnostic only when loss_state=0
predicted_visual_future: [400, 768]
```

其中 `latent_action` 是后续接入 LaWM decoder、FutureContactDiT、触觉/力觉 adapter 的中间动作表示。当前最好的 teacher 是 visual-only v4：

```text
checkpoint:
  checkpoints/latent_action_idm_droid100_visual_teacher_v4/best.pt

config:
  latent_action_idm/configs/dino_idm_droid100_visual_teacher_v4.yaml
```

单模块 smoke test：

```bash
python -m latent_action_idm.scripts.smoke_test_stage1_modules \
  --batch-size 2 \
  --num-tokens 400 \
  --token-dim 768 \
  --state-dim 7 \
  --latent-dim 128 \
  --hidden-dim 128 \
  --layers 1 \
  --heads 4 \
  --device cpu
```

输出应包含：

```text
IDM latent_mu: (2, 128)
LaWM predicted future: (2, 400, 768)
Stage1 predicted_state_future: (2, 7)
```

## DROID100 v4 Teacher Benchmark

当前 droid_100 实验使用：

```text
dataset: lerobot/droid_100
episodes: 100
frames: 32212
fps: 15
future horizon: 15 frames = 1 second
train windows: 3107
val windows: 777
visual tokens: [400, 768]
```

v4 teacher 结构：

```text
Frozen DINOv3 ViT-B/16
  -> normalized two-view visual tokens
  -> camera-aware VisualTokenProjector
  -> visual-only IDM q(z | u_t, u_T)
  -> z_teacher [128]
  -> residual LaWM decoder: u_T_hat = u_t + delta_u_hat
```

Benchmark 命令：

```bash
python -m latent_action_idm.scripts.benchmark_idm_teacher \
  --checkpoint checkpoints/latent_action_idm_droid100_visual_teacher_v4/best.pt \
  --train-manifest data/manifests/droid100_train.jsonl \
  --val-manifest data/manifests/droid100_val.jsonl \
  --output outputs/idm_teacher_benchmark/droid100_visual_teacher_v4.txt \
  --batch-size 8 \
  --device cuda
```

结果：

```text
identity:
  future_mse_per_token: 0.04973803
  future_improvement_vs_identity: 0.00000000
  transition_delta_cosine: 0.00000000
  retrieval_top1: 0.02316602
  retrieval_top5: 0.45817246
  retrieval_top10: 0.63577864
  retrieval_median_rank: 6

mean_delta:
  future_mse_per_token: 0.04971497
  future_improvement_vs_identity: 0.00046351
  transition_delta_cosine: 0.02211529
  retrieval_top1: 0.02445302

nearest_neighbor:
  future_mse_per_token: 0.11024701
  future_improvement_vs_identity: -1.21655370
  transition_delta_cosine: 0.32939101
  retrieval_top1: 0.03603604

teacher v4:
  future_mse_per_token: 0.03619018
  future_improvement_vs_identity: 0.27238409
  delta_r2: 0.27238021
  transition_delta_cosine: 0.50969794
  retrieval_top1: 0.50579151
  retrieval_top5: 0.92792793
  retrieval_top10: 0.95752896
  retrieval_median_rank: 1
```

结论：

```text
v4 teacher 明显优于 identity / mean-delta / nearest-neighbor baseline。
它可以作为后续 frozen-IDM DiT-LaWM 的 z_teacher 来源。
```

## 后续创新修改路线

复现 LaWAM Stage 1 后，再加入本项目的触觉模块：

```text
DINOv3 visual patch tokens
UR5 state
tactile / force / gripper feedback
  -> Contact-Gated IDM / LaWM
  -> latent_action
  -> future visual-contact latent
  -> damage / slip / deformation outcome
```

推荐阶段：

```text
1. 当前 patch-token Transformer IDM/LaWM 跑通 60 条数据
2. 将 encoder_layers / decoder_layers 提升到 LaWAM 复现规模
3. 加入 tactile-force encoder 和 contact gate
4. 加入 damage/slip/deformation heads
```

## DiT-LaWM 实验路线

如果要验证 diffusion Transformer 作为 LaWM decoder，使用独立配置和训练入口：

```bash
python -m latent_action_idm.scripts.smoke_test_dit_lawam \
  --batch-size 2 \
  --num-tokens 400 \
  --token-dim 768 \
  --state-dim 7 \
  --latent-dim 128 \
  --hidden-dim 128 \
  --layers 1 \
  --heads 4 \
  --device cpu
```

训练：

```bash
python -m latent_action_idm.train_dit_lawam \
  --config latent_action_idm/configs/dit_lawam.yaml
```

DiT-LaWM 输入输出：

```text
clean future tokens u_T:       [B, 400, 768]
random noise epsilon:          [B, 400, 768]
diffusion timestep tau:        [B]
noisy future tokens x_tau:     [B, 400, 768]
current visual tokens u_t:     [B, 400, 768]
latent action z from IDM:      [B, 128]

DiffusionLatentWorldModel:
  x_tau + u_t + z + tau
    -> predicted noise epsilon_hat [B, 400, 768]
```

训练目标：

```text
L_dit = MSE(epsilon_hat, epsilon)
```

完整 Stage1DiTLaWAM：

```text
u_t + u_T
  -> frozen visual-only IDM v4
  -> z_teacher

u_T + noise -> x_tau

x_tau + u_t + z_teacher + tau
  -> DiffusionLatentWorldModel
  -> predicted noise
  -> predicted future tokens
```

建议实验顺序：

```text
1. 你的 60 episodes:
   验证代码、显存、loss 是否下降。

2. lerobot/droid_100:
   小规模 sanity check。

3. lerobot/droid_1.0.1:
   正式验证 DiT-LaWM 是否优于 deterministic LaWM。
```

## Future Token Evaluator

当前有两类 evaluator：

```text
1. metric_future_scores
   无监督度量，不需要训练，直接比较 u_T_hat 和真实 u_T。

2. FutureTokenEvaluator
   可训练网络，后续有 task success / goal progress / risk 标签后使用。
```

FutureTokenEvaluator 输入输出：

```text
输入:
  visual_t:                  [B, 400, 768]
  predicted_visual_future:   [B, 400, 768]
  state_t optional:          [B, 7]
  latent_action optional:    [B, 128]

输出:
  goal_progress: [B]
  success_logit: [B]
  risk_logit:    [B]
  value:         [B]
```

smoke test：

```bash
python -m latent_action_idm.scripts.smoke_test_future_evaluator \
  --batch-size 2 \
  --num-tokens 400 \
  --token-dim 768 \
  --state-dim 7 \
  --latent-dim 128 \
  --hidden-dim 128 \
  --layers 1 \
  --heads 4 \
  --device cpu
```

## DiT-LaWM Evaluation

训练 DiT 后，用这个脚本评估 clean future token 质量：

```bash
python -m latent_action_idm.scripts.analyze_dit_lawam \
  --checkpoint checkpoints/dit_lawam/best.pt \
  --manifest data/manifests/idm_val.jsonl \
  --split-name val \
  --output-dir outputs/dit_lawam \
  --batch-size 4 \
  --timestep 500 \
  --device cuda
```

输出：

```text
outputs/dit_lawam/val.metrics.txt
outputs/dit_lawam/val.csv
outputs/dit_lawam/val.npz
outputs/dit_lawam/val_base_patch_mse_14x14.csv
outputs/dit_lawam/val_wrist_patch_mse_14x14.csv
```

重点指标：

```text
future_mse_per_token
identity_future_mse_per_token
future_improvement_vs_identity
noise_mse_per_token
transition_delta_cosine
state_improvement_vs_identity
```

判断标准：

```text
future_improvement_vs_identity > 0
  说明 DiT-LaWM 预测的 future tokens 比直接使用 u_t 更接近真实 u_T。

transition_delta_cosine 越高
  说明预测的视觉变化方向越接近真实视觉变化方向。

patch_mse_14x14
  用于定位 base / wrist 视角中哪些区域的未来预测误差最大。
```
