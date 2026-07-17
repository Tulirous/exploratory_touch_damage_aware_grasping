# UR5 + Linker Hand O6 Human Mimic Demo

固定 D435 观察操作者上半身和右手，UR5 只跟随右手腕三维平移，右手
Linker Hand O6 跟随手指协同动作。第一阶段只做无接触空中模仿；物体抓取、
压力闭环属于第二阶段。

## 当前边界

- `synthetic` tracker：不需要相机，用于验证全链路；
- `realsense` tracker：D435 深度提供腕部米制位置，MediaPipe 提供无标记右手
  21 点；
- `ManoForwardAdapter`：将完整 45 维 MANO pose 前向为相同的 21 点接口；
- `dry-run`：运行所有跟踪、重定向、安全和日志逻辑，但不发送硬件命令；
- `hardware`：通过 `ur_rtde` 和 Linker Hand Python SDK 控制实机。

MediaPipe 后端不是 MANO 参数估计器。它只让第一版 Demo 在没有额外模型权重时
立即可运行。以后接入 HaMeR/Dyn-HaMR 时，只需使新 tracker 输出
`TrackingSample(wrist_xyz_m, hand_joints)`，控制层无需修改。

## 安装

最小 dry-run：

```bash
python3 -m pip install -r human_mimic_demo/requirements-core.txt
```

D435：

```bash
python3 -m pip install -r human_mimic_demo/requirements-camera.txt
python3 -m human_mimic_demo.scripts.download_mediapipe_models
```

### 当前本机的 D435 状态

本机是 Apple Silicon macOS，USB 系统已经枚举到：

```text
Intel(R) RealSense(TM) Depth Camera 435
Vendor ID: 0x8086
Product ID: 0x0b07
```

因此相机连接和 USB 枚举正常；当前缺少的是 Python 运行时中的
`pyrealsense2`、`opencv-python` 和 `mediapipe`。RealSense 官方将 macOS 标为
“可编译但未验证”，Apple Silicon 环境不应假定 PyPI 一定提供兼容 wheel。如果
`pip install pyrealsense2` 没有匹配发行包，需要从 librealsense 源码启用
`BUILD_PYTHON_BINDINGS` 编译，或者在 Ubuntu 20.04/22.04/24.04 主机运行相机进程。
正式 UR5/O6 Demo 推荐 Ubuntu，因为 RealSense、`ur_rtde` 和 Linker Hand CAN SDK
可以放在同一台控制机上，部署风险最低。

### Windows 主机部署

阶段 A 可以直接使用原生 Windows 10/11 x64，不要求 Ubuntu，也不要放在 WSL2 或
虚拟机中。D435 官方提供 Windows SDK/Python 包；`ur_rtde` 发布了 Windows x64
Python wheels；Linker Hand Python SDK 的 CAN 配置也区分 Windows 通道。

建议使用 Python 3.11 或 3.12 x64：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r human_mimic_demo\requirements-camera.txt
python -m pip install -r human_mimic_demo\requirements-hardware.txt
python -m human_mimic_demo.scripts.download_mediapipe_models
```

先安装并运行 RealSense Viewer，分别确认 RGB 与 depth 稳定，再运行只读 Demo：

先执行不涉及机器人和灵巧手的 D435 检测：

```powershell
python -m human_mimic_demo.scripts.check_d435 --seconds 5 --preview
```

成功时应显示 `PASS`，并生成
`human_mimic_demo\logs\d435_check.json`。如果 RealSense Viewer 正在占用相机，
请先关闭 Viewer 再运行检测脚本。

相机检测通过后再运行只读 Demo：

```powershell
python -m human_mimic_demo.app `
  --config human_mimic_demo\configs\demo_windows.json `
  --tracker realsense `
  --mode dry-run `
  --display
```

Windows 的 Linker Hand CAN 通道取决于 USB-CAN 盒：官方配置说明中，蓝色盒通常为
`PCAN_USBBUS1`，透明盒通常为 `0`。必须先用 Linker Hand 官方示例确认通道，再修改
`demo_windows.json`；不要在未确认通道和 O6 范围时启动 hardware 模式。

Windows 主机还需要通过有线以太网访问 UR5，Windows 防火墙需允许 Python 的专用
网络通信。D435 继续连接 Windows USB 3 端口，USB-CAN 也连接同一主机。

实机另外安装 `ur_rtde`，并克隆官方 SDK：

```bash
git clone https://github.com/linker-bot/linkerhand-python-sdk.git
python3 -m pip install -r human_mimic_demo/requirements-hardware.txt
```

将 `configs/demo.json` 中的 `hand.sdk_path` 改为 SDK 仓库的绝对路径。

## 运行

### 1. 无硬件 smoke test

```bash
python3 -m human_mimic_demo.app \
  --tracker synthetic \
  --mode dry-run \
  --auto-calibrate \
  --auto-arm \
  --duration 5
```

### 2. D435 只读模式

```bash
python3 -m human_mimic_demo.app \
  --tracker realsense \
  --mode dry-run \
  --display
```

窗口按键：

| 按键 | 功能 |
|---|---|
| `c` | 当前人腕与当前 UR5 TCP 设为相对零点 |
| `o` | 记录当前人手为张开姿态 |
| `f` | 记录当前人手为握拳姿态 |
| `a` | 使能/解除控制 |
| `e` | 软件急停，本进程内不可恢复 |
| `q` | 退出 |

先在只读模式观察日志中的 `target_tcp` 和 `o6_command`。移动人手时逐轴确认：

1. 人手向右，机器人目标沿预期方向变化；
2. 人手向上，机器人目标沿预期方向变化；
3. 人手靠近/远离相机，机器人目标沿预期方向变化。

如果方向不对，只修改 `camera_to_robot_rotation`。该矩阵必须是正交矩阵；不要用
平移缩放掩盖轴向错误。

### 3. 实机模式

实机前必须修改：

- UR5 IP；
- 机器人真实安全工作空间；
- D435 到 UR5 base 的轴向映射；
- O6 每轴张开、闭合、安全上下限；
- O6 低速和低力矩设置。

确保周围无人、UR5 示教器急停可触达，并且没有 ROS、手套或其他程序同时控制
O6。然后运行：

```bash
python3 -m human_mimic_demo.app \
  --tracker realsense \
  --mode hardware \
  --display \
  --i-understand-hardware-risk
```

实机模式刻意禁止 `--auto-calibrate` 和 `--auto-arm`。操作顺序必须是
`o`、`f`、`c`，检查画面与初始姿态，最后按 `a`。

## 验证

```bash
python3 -m unittest discover -s human_mimic_demo/tests -v
python3 -m py_compile \
  human_mimic_demo/app.py \
  human_mimic_demo/safety.py \
  human_mimic_demo/retargeting/o6.py \
  human_mimic_demo/tracking/realsense_mediapipe.py \
  human_mimic_demo/controllers/ur5.py \
  human_mimic_demo/controllers/linkerhand_o6.py
```

运行日志保存在 `human_mimic_demo/logs/session_*.jsonl`，该目录不应作为真实验证
证据，除非其中确实来自 D435 和实机控制。
