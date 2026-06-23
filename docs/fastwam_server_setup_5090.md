# Fast-WAM 服务器部署记录：RTX 5090

本文件用于在 RTX 5090 服务器上跑通 Fast-WAM 官方环境。目标不是训练本课题模型，而是先确认：

```text
Fast-WAM 官方仓库可安装
PyTorch 能识别 RTX 5090
Wan2.2 / ActionDiT 预处理能跑
released checkpoint 可下载
```

## 0. 本机状态

服务器 Fast-WAM 仓库路径：

```text
~/fastwam
```

服务器 Fast-WAM 相关数据根目录：

```text
/data/chy/fastwam
```

官方仓库：

```text
https://github.com/yuantianyuan01/FastWAM.git
```

## 1. 服务器先检查

登录服务器后先执行：

```bash
nvidia-smi
python3 --version
conda --version
df -h
```

RTX 5090 是 Blackwell 架构。Fast-WAM 官方依赖 PyTorch `2.7.1+cu128`，CUDA 12.8 对 Blackwell 支持更稳，因此驱动需要足够新。若 `nvidia-smi` 显示的 CUDA Version 低于 12.8，优先让管理员升级驱动。

你当前服务器状态：

```text
GPU: NVIDIA GeForce RTX 5090 D, 32607 MiB
Driver: 595.45.04
CUDA Version shown by nvidia-smi: 13.2
```

这说明驱动足够新。Fast-WAM 官方安装的是 PyTorch `2.7.1+cu128`，这里的 `cu128` 是 PyTorch wheel 自带 CUDA runtime 版本，不要求系统 `nvidia-smi` 显示完全等于 12.8；只要驱动能兼容即可。

## 1.1 数据集应该放在哪里

当前 `df -h` 显示：

```text
/       1006G, available 652G
/data     14T, available 11T
/data1   9.7T, available 7.3T
/data2   3.8T, available 3.6T
```

建议：

```text
代码仓库:    ~/fastwam
conda 环境:  默认放在 home 或 conda 的 envs 目录
大数据集:    /data/chy/fastwam/datasets
模型权重:    /data/chy/fastwam/checkpoints
实验输出:    /data/chy/fastwam/runs
```

优先把 LIBERO、RoboTwin、Wan/Fast-WAM checkpoint 放到 `/data`，因为它剩余空间最大。不要把大数据集直接放到 `/` 或 home 下，避免系统盘被占满。

推荐创建目录：

```bash
mkdir -p /data/chy/fastwam/datasets
mkdir -p /data/chy/fastwam/checkpoints
mkdir -p /data/chy/fastwam/runs
```

如果 Fast-WAM 代码默认读仓库内的 `data/`、`checkpoints/`、`runs/`，可以在仓库里用软链接指向 `/data`：

```bash
cd ~/fastwam
ln -sfn /data/chy/fastwam/datasets data
ln -sfn /data/chy/fastwam/checkpoints checkpoints
ln -sfn /data/chy/fastwam/runs runs
```

这样代码路径仍然是：

```text
FastWAM/data
FastWAM/checkpoints
FastWAM/runs
```

但实际文件写入 `/data`，更适合大数据。

后续查看空间：

```bash
df -h
du -sh /data/chy/fastwam/datasets
du -h --max-depth=1 /data/chy/fastwam/datasets | sort -h
```

## 2. Clone 仓库

建议放在服务器自己的工作目录，例如：

```bash
cd ~/projects
git clone https://github.com/yuantianyuan01/FastWAM.git
cd FastWAM
```

## 3. 创建环境

官方 README 使用 Python 3.10：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
```

安装 PyTorch CUDA 12.8：

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

安装 Fast-WAM：

```bash
pip install -e .
```

## 4. 验证 PyTorch 和 5090

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
print("ok", y.shape, y.dtype)
PY
```

期望：

```text
available: True
device: NVIDIA GeForce RTX 5090
capability: Blackwell 对应的 12.x
```

如果这里失败，不要继续跑 Fast-WAM。

## 5. 准备 Wan / ActionDiT

Fast-WAM 官方配置：

```text
model_id: Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
video_dit hidden_dim: 3072
action_dit hidden_dim: 1024
```

先设置 checkpoint 目录：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

如果已经按上面的软链接方案把 `checkpoints` 指到 `/data/chy/fastwam/checkpoints`，这个环境变量仍然保持：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

因为 `$(pwd)/checkpoints` 实际会写入 `/data/chy/fastwam/checkpoints`。

生成 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

这一步会触发模型下载/加载，显存和网络都要正常。

## 6. 下载 released checkpoints

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

先不急着下载 LIBERO/RoboTwin 全数据。released checkpoint 下载成功后，下一步再决定跑 LIBERO 还是只做本课题 latent extraction。

如果新版 `huggingface_hub` 提示 `huggingface-cli` 已废弃，改用：

```bash
hf download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --repo-type model \
  --local-dir /data/chy/fastwam/checkpoints/fastwam_release
```

如果服务器无法直连 Hugging Face，可先尝试镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果 `hf download` 仍报 `Local entry not found` 或远端资源不可达，先用 `wget` 验证单个小文件：

```bash
cd /data/chy/fastwam/checkpoints/fastwam_release
wget -c https://hf-mirror.com/yuanty/fastwam/resolve/main/libero_uncond_2cam224_dataset_stats.json
```

这个 JSON 能下载后，再继续下载 `.pt` 权重：

```bash
wget -c https://hf-mirror.com/yuanty/fastwam/resolve/main/libero_uncond_2cam224.pt
wget -c https://hf-mirror.com/yuanty/fastwam/resolve/main/robotwin_uncond_3cam_384.pt
wget -c https://hf-mirror.com/yuanty/fastwam/resolve/main/robotwin_uncond_3cam_384_dataset_stats.json
```

当前服务器已完成 LIBERO released checkpoint 下载：

```text
/data/chy/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224.pt
/data/chy/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

RoboTwin released checkpoint 可以后续再下载；短期先跑通 LIBERO。

## 6.1 安装 LIBERO eval 依赖

如果运行：

```bash
python experiments/libero/run_libero_manager.py --help
```

报错：

```text
ModuleNotFoundError: No module named 'libero'
```

说明当前 `fastwam` 环境还没有安装 LIBERO 官方包。LIBERO 官方 README 的原始环境使用 Python 3.8 和旧版 PyTorch，但 Fast-WAM 需要 PyTorch `2.7.1+cu128`。因此不要在 `fastwam` 环境里安装 LIBERO README 中的旧 PyTorch，只安装 LIBERO 包和仿真依赖。

建议：

```bash
cd /data/chy/fastwam
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .

cd ~/fastwam
pip install mujoco==3.3.2
```

安装后验证：

```bash
python - <<'PY'
from libero.libero import benchmark
print("libero ok")
print(list(benchmark.get_benchmark_dict().keys())[:10])
PY
```

如果安装后仍然报：

```text
ModuleNotFoundError: No module named 'libero'
```

优先检查当前 `python` 和 `pip` 是否属于同一个 conda 环境：

```bash
which python
which pip
python -m pip --version
python -m pip show libero
```

如果 `python -m pip show libero` 没有输出，说明 LIBERO 没有装进当前环境。重新用当前解释器安装：

```bash
cd /data/chy/fastwam/LIBERO
python -m pip install -r requirements.txt
python -m pip install -e .
```

Fast-WAM 的 LIBERO manager 使用：

```python
from libero.libero import benchmark
```

因此需要把 LIBERO 仓库外层目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/data/chy/fastwam/LIBERO:$PYTHONPATH
```

首次 import LIBERO 时会尝试交互式创建 `~/.libero/config.yaml`。在脚本或 heredoc 里运行时可能报：

```text
Do you want to specify a custom path for the dataset folder? (Y/N):
EOFError: EOF when reading a line
```

可手动创建非交互配置：

```bash
mkdir -p ~/.libero
mkdir -p /data/chy/fastwam/datasets/libero
cat > ~/.libero/config.yaml <<'YAML'
assets: /data/chy/fastwam/LIBERO/libero/libero/assets
bddl_files: /data/chy/fastwam/LIBERO/libero/libero/bddl_files
benchmark_root: /data/chy/fastwam/LIBERO/libero/libero
datasets: /data/chy/fastwam/datasets/libero
init_states: /data/chy/fastwam/LIBERO/libero/libero/init_files
YAML
```

如果手动运行 `experiments/libero/eval_libero_single.py` 时在：

```text
LIBERO/libero/libero/benchmark/__init__.py
task_suite.get_task_init_states(...)
torch.load(init_states_path)
```

报 PyTorch 2.6+ 的：

```text
_pickle.UnpicklingError: Weights only load failed
```

这是因为 PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，而 LIBERO 的 init state 文件不是纯权重。由于 init state 文件来自官方 LIBERO 仓库，可在本地 LIBERO 源码中把：

```python
init_states = torch.load(init_states_path)
```

改成：

```python
init_states = torch.load(init_states_path, weights_only=False)
```

当前服务器已完成一次 LIBERO 单任务验证：

```text
task_suite: libero_10
task_id: 0
task_description: put both the alphabet soup and the tomato sauce in the basket
successes: 1
total_episodes: 1
duration: 108.68 s
video: evaluate_results/libero/manual_libero10_task0_debug/libero_10/videos/*.mp4
```

这说明官方 released checkpoint、LIBERO 仿真、Fast-WAM policy forward 和视频保存链路已经跑通。后续应先扩展到少量 trials / 少量 task，而不是直接追求完整论文表格。

补充验证：

```text
libero_10 task_id=0: 5/5 success, duration 169.26 s
libero_10 task_id=1: 1/1 success, duration 105.84 s
```

这可以作为服务器上官方 Fast-WAM released checkpoint 可运行的短期复现证据。

## 7. 单卡 5090 的现实边界

RTX 5090 通常可用于：

```text
官方代码安装和推理验证
ActionDiT 预处理
小规模 latent extraction
本课题 adapter / outcome head 训练
```

不建议在单张 5090 上做：

```text
严格复现 Fast-WAM 训练
RoboTwin 规模训练
从零训练 Wan2.2-5B
```

Fast-WAM 官方 README 写明：

```text
LIBERO: 单节点 8 GPUs
RoboTwin: 64 GPUs 加速训练
```

所以服务器 5090 的合理目标是 **跑通官方环境 + 提取 latent + 本课题微调**，不是复现完整训练。

## 8. 和本课题的连接

服务器跑通 Fast-WAM 后，下一步接本项目：

```text
GELLO / LeRobot base+wrist video
  -> Fast-WAM/Wan frozen backbone
  -> visual_latent.npy
  -> 回到 exploratory_touch_damage_aware_grasping/train.py
```

本项目的 latent extraction 接口在：

```text
scripts/extract_visual_latents.py
```

当前 `--backbone dummy` 只用于测试 shape。正式接入时，需要把 Fast-WAM/Wan 的视频编码逻辑补进这个脚本，或者单独写一个服务器端 latent extraction 脚本后把 `.npy` 复制回本项目。
