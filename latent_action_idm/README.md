# LaWAM-Aligned Latent Action IDM

这个子项目用于复现 LaWAM Stage 1 的核心思想，并作为后续加入触觉/力觉模块的起点。当前目标不是直接训练 VLA，也不是生成未来 RGB 视频，而是学习：

```text
D435/D415 当前视觉 latent + D435/D415 未来视觉 latent + 当前 UR5 state
  -> latent_action
  -> 未来 UR5 state / 未来视觉 latent
```

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
  输入: visual_t, visual_future, state_t
  输出: latent_mu, latent_logvar, latent_action

latent_action_idm/models/latent_world_model.py
  LatentWorldModelDecoder
  输入: visual_t, latent_action
  输出: predicted_visual_future

latent_action_idm/models/stage1_lawam.py
  Stage1LaWAM
  组合: IDM + LaWM decoder + auxiliary state predictor
```

为了适配双相机，当前默认把两个视角的 patch tokens 沿 token 维拼接：

```text
base image:  14 x 14 = 196 tokens
wrist image: 14 x 14 = 196 tokens
two views:   392 tokens
token dim:   768
```

训练配置默认使用 8 层 Transformer encoder/decoder，方便先在 60 条数据上调通。正式复现 LaWAM 规模时，把配置中的层数调高：

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
visual_t: [392, 768]
visual_future: [392, 768]
state_t: [7]
state_future: [7]
```

模型输出：

```text
latent_action: [128]
predicted_state_future: [7]
predicted_visual_future: [392, 768]
```

其中 `latent_action` 是后续接入 LaWM decoder、FutureContactDiT、触觉/力觉 adapter 的中间动作表示。

单模块 smoke test：

```bash
python -m latent_action_idm.scripts.smoke_test_stage1_modules \
  --batch-size 2 \
  --num-tokens 392 \
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
LaWM predicted future: (2, 392, 768)
Stage1 predicted_state_future: (2, 7)
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
  --num-tokens 392 \
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
clean future tokens u_T:       [B, 392, 768]
random noise epsilon:          [B, 392, 768]
diffusion timestep tau:        [B]
noisy future tokens x_tau:     [B, 392, 768]
current visual tokens u_t:     [B, 392, 768]
latent action z from IDM:      [B, 128]

DiffusionLatentWorldModel:
  x_tau + u_t + z + tau
    -> predicted noise epsilon_hat [B, 392, 768]
```

训练目标：

```text
L_dit = MSE(epsilon_hat, epsilon)
```

完整 Stage1DiTLaWAM：

```text
u_t + u_T + state_t
  -> InverseDynamicsTransformer
  -> z

u_T + noise -> x_tau

x_tau + u_t + z + tau
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
