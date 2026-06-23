# Fast-WAM Visual Latent Adapter

本文档说明第一版 Fast-WAM 迁移代码。目标是让官方 Fast-WAM 为本项目提供 frozen visual latent，而不是让 Fast-WAM 直接控制 UR5。

## 用到 Fast-WAM 的部分

```text
D435 base RGB frame + D415 wrist RGB frame
  -> center-crop resize 到 224x224
  -> 横向拼接成 224x448
  -> normalize 到 [-1, 1]
  -> Fast-WAM/Wan VAE image encoder
  -> Fast-WAM video_expert.pre_dit
  -> mean-pool video_pre["tokens"]
  -> visual_latent.npy, shape (3072,)
```

对应官方 Fast-WAM 代码位置：

```text
experiments/libero/libero_utils.py
  get_libero_image()

experiments/libero/eval_libero_single.py
  _obs_to_model_input()
  _predict_action_chunk()

src/fastwam/models/wan22/fastwam.py
  _encode_input_image_latents_tensor()
  video_expert.pre_dit()
```

## 没有用到 Fast-WAM 的部分

第一版不使用：

```text
Fast-WAM action diffusion policy 直接输出真实 UR5 动作
LIBERO / RoboTwin action space
未来视频生成作为推理必要步骤
官方 Fast-WAM 完整训练流程
从零训练 Wan2.2-5B / ActionDiT / 通用 WAM
```

本项目自己的模型仍然负责：

```text
visual_latent + tactile_force_latent + candidate_action_latent
  -> success / slip / damage / deformation / recovery_ratio
```

## 新增代码

```text
datasets/video_pair_reader.py
  读取 base/wrist 视频的同步 RGB 帧

models/visual_backbone_adapters/fastwam_wan_adapter.py
  动态导入官方 FastWAM，加载 released checkpoint，抽取 pooled video tokens

scripts/extract_fastwam_visual_latents.py
  读取本项目 JSONL manifest，批量保存 visual_latent.npy
```

## 服务器运行示例

在服务器上把本项目上传后执行：

```bash
cd /path/to/exploratory_touch_damage_aware_grasping
conda activate fastwam

export DIFFSYNTH_MODEL_BASE_PATH=/data/chy/fastwam/checkpoints
export PYTHONPATH=/path/to/exploratory_touch_damage_aware_grasping

python scripts/extract_fastwam_visual_latents.py \
  --manifest data/manifests/train.jsonl \
  --fastwam-root /home/chy/fastwam \
  --ckpt /data/chy/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --device cuda \
  --mixed-precision bf16 \
  --frame-index 0 \
  --overwrite
```

第一版只抽取每条 episode 的单帧视觉 latent。后续如果需要利用轻触窗口的时序信息，再扩展为多帧抽取并把输出 shape 从 `(3072,)` 升级为 `(T, 3072)`。
