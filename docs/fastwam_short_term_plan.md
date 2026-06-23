# Fast-WAM 短期执行计划

本项目当前短期目标：

```text
先复现官方 Fast-WAM 的可运行链路
  -> 理解 Fast-WAM/Wan latent backbone
  -> 接入本项目 UR5 + GELLO + RGB-D + 触觉/力觉 + 易损标签
  -> 训练 Fast-WAM-style latent outcome prediction
```

这里的“复现 Fast-WAM”不是指单卡从头训练出论文完整结果。官方 Fast-WAM 使用 Wan2.2-5B video DiT 和 ActionDiT，LIBERO 训练参考为单节点 8 GPU，RoboTwin 训练参考为 64 GPU。RTX 5090D 单卡的合理目标是环境、released checkpoint、推理/评估、latent extraction 和本项目 adapter/outcome head 训练。

## 阶段 0：服务器与存储准备

已验证服务器条件：

```text
GPU: RTX 5090 D 32GB
Driver: 595.45.04
CUDA shown by nvidia-smi: 13.2
大容量磁盘: /data 约 11T 可用
```

项目长期固定环境：

```text
本地采集/闭环客户端: RTX 4070 8GB 笔记本
云端最低显存: 24GB，用于 outcome head / adapter 训练和小批量 latent extraction
云端推荐显存: 32GB，用于 Fast-WAM released checkpoint 推理和 frozen latent extraction
云端更稳显存: 40GB-80GB，用于长窗口、更大 batch、LoRA / adapter 微调
```

推荐布局：

```text
~/fastwam                          官方 Fast-WAM 代码
/data/chy/fastwam/datasets         LIBERO / RoboTwin / 后续本项目视频数据
/data/chy/fastwam/checkpoints      Wan / Fast-WAM / ActionDiT checkpoints
/data/chy/fastwam/runs             评估和训练输出
```

在官方 Fast-WAM 仓库里用软链接保持默认路径：

```bash
cd ~/fastwam
ln -sfn /data/chy/fastwam/datasets data
ln -sfn /data/chy/fastwam/checkpoints checkpoints
ln -sfn /data/chy/fastwam/runs runs
```

## 阶段 1：跑通官方 Fast-WAM 环境

目标：

```text
PyTorch 能识别 RTX 5090D
Fast-WAM 官方仓库可安装
Wan / ActionDiT 预处理可执行
released checkpoint 可下载
```

命令记录见：

```text
docs/fastwam_server_setup_5090.md
```

优先验证：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

然后按官方 README 下载 checkpoint，先不急着下载完整 RoboTwin 大数据。

## 阶段 2：跑 released checkpoint

优先顺序：

```text
1. 下载 Fast-WAM released checkpoints 和 dataset_stats。
2. 选择 LIBERO 作为第一个复现实验，因为规模和环境复杂度通常小于 RoboTwin。
3. 先跑很小的评估子集或单任务，确认推理链路可运行。
4. 再决定是否下载完整 LIBERO / RoboTwin 数据。
```

判断通过标准：

```text
官方 checkpoint 能加载
evaluation manager 能启动
单卡配置下能完成至少一个小规模推理任务
显存不会在模型加载阶段 OOM
```

不要把单卡小规模运行结果表述为完整复现论文结果。

当前进度：

```text
LIBERO released checkpoint 已下载。
ActionDiT backbone 已生成。
libero_10 task_id=0, 5 trials 已完成，successes=5/5，并保存 rollout mp4。
libero_10 task_id=1, 1 trial 已完成，successes=1/1。
```

这可以证明官方 Fast-WAM released checkpoint 在服务器上可运行，但还不是完整论文结果复现。

## 阶段 3：理解并抽取 visual/video latent

本项目要复用的是：

```text
base_video + wrist_video
  -> frozen Fast-WAM/Wan latent backbone
  -> visual_latent.npy
```

需要在官方仓库中确认：

```text
视频输入预处理尺寸、帧数、归一化方式
Wan / Fast-WAM backbone 输出 token 或 hidden state 的位置
latent shape 是否可压缩到 configs/fastwam_fragile.yaml 的 visual_latent_dim=3072
是否需要按 episode 保存一个 pooled latent，还是保存时序 latent
```

第一版建议输出每条 episode 一个定长 latent：

```text
data/latents/<episode_id>_visual.npy
shape: (3072,)
```

如果保留时序 token，则需要同步修改本项目 dataset 和 model，目前默认先不扩展。

## 阶段 4：接入本项目数据

本项目数据流保持：

```text
GELLO 遥操作
  -> LeRobot V3.0 episode
  -> pressure / force / gripper sidecar
  -> success / slip / damage / deformation labels
  -> JSONL manifest
  -> visual_latent.npy
  -> train.py
```

短期最小数据目标：

```text
豆腐 / 海绵等 50-100 条 episode
每条 episode 至少有 base video、wrist video、force、gripper、tactile、executed_action、damage/slip/success 标签
candidate action 第一版可设 K=1，即 executed_action
```

训练目标：

```text
visual_latent + tactile_force_latent + candidate_action_latent
  -> success / slip / damage / deformation / recovery_ratio
```

## 阶段 5：短期验收标准

可以认为短期目标完成，当满足：

```text
官方 Fast-WAM 环境在 RTX 5090D 上跑通
released checkpoint 可加载并完成小规模推理
本项目至少一批 GELLO episode 能生成 manifest
真实或半真实 visual_latent.npy 能被 train.py 读取
FastWAM-Fragile outcome model 能完成训练循环
```

如果没有真实易损物体数据，只能说明 pipeline 跑通，不能声称 damage/slip 模型已经验证。
