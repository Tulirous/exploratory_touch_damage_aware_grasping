# 技术设计：无标记人体手腕与右手动作模仿

## 1. 需求冻结

阶段 A：固定 D435 看到操作者上半身、完整右臂和右手，无标记跟踪；UR5 只模仿
右手腕三维平移，TCP 姿态保持标定时不变；Linker Hand O6 模仿右手手型。

阶段 B：在阶段 A 稳定后增加人侧与机器人侧物体抓取、接触检测及损伤感知控制。

目标指标：15–30 Hz，端到端延迟小于 200 ms；跟踪丢失 350 ms 后保持；所有运动
限制在人手相对范围、单周期位移和机器人绝对安全盒内。

## 2. 数据流

```text
D435 RGB                    D435 depth
   │                             │
   └─ right-hand 21 landmarks    └─ metric wrist XYZ
                  │                  │
                  └──── TrackingSample ────┐
                                           │
                  ┌────────────────────────┴──────────────┐
                  │                                       │
          relative wrist mapping                   O6 retargeting
          camera frame -> UR base                  21 joints -> 6 synergies
                  │                                       │
          workspace/step limiter                   filter/range limiter
                  │                                       │
             ur_rtde servoL                         finger_move
```

人腕映射只使用相对量：

```text
delta_h = wrist_now_camera - wrist_origin_camera
delta_r = R_camera_to_robot @ (scale * delta_h)
tcp_xyz = tcp_origin_xyz + clipped(delta_r)
```

TCP 的旋转向量始终复制 `tcp_origin[3:6]`，因此视觉姿态抖动不会传给 UR5。

## 3. 为什么控制层使用 21 点接口

MANO 完整 hand pose 是 15 个局部关节的 45 维轴角，SMPL-X/H 也可能输出 PCA
手部姿态。直接把这些数值映射到 O6 会混淆坐标轴、模型 rest pose 和机械结构。

所有后端先前向到一个统一的 21 点右手几何接口，再提取：

```text
[thumb flexion, thumb abduction,
 index flexion, middle flexion, ring flexion, pinky flexion]
```

O6 是六路协同驱动，无法复现 MANO 的完整自由度。首版保留五指屈曲和拇指对掌
关系。第二版可使用 O6 URDF 或实机拟合的 `q -> fingertips` 响应模型，最小化
归一化指尖位置和指尖间距误差。

## 4. 跟踪后端

### 当前设备环境

开发机为 Apple Silicon macOS，D435 已通过 USB 正确枚举。当前 Python 环境尚未
安装 RealSense、OpenCV 和 MediaPipe 包。macOS 路线需要验证从源码编译的
librealsense Python bindings；正式实机控制优先部署到 Ubuntu 控制机，代码和配置
格式不变。

阶段 A 也支持原生 Windows 10/11 x64 单机部署：D435、O6 USB-CAN 和 UR5 Ethernet
全部连接 Windows。Windows 不具备 Linux 实时内核保证，因此采用保守的 15–25 Hz、
低速度、小工作空间验证。阶段 B 的接触/损伤闭环和正式重复实验仍优先使用原生
Ubuntu；不建议通过 WSL2/虚拟机转发 D435 或 CAN。

### 当前实时后端

MediaPipe Tasks Hand Landmarker 负责无标记右手 21 点；D435 在手腕像素邻域做深度中值，再利用相机内参
反投影到米制 XYZ。手型只使用相对 21 点，因此不受腕部绝对深度影响。

新版 MediaPipe 已移除 `mp.solutions`；实时后端使用 VIDEO 模式的 Tasks API，并从
官方模型仓库单独下载 `hand_landmarker.task`。当前 Windows D435 实测以 MediaPipe
的 `Right` 标签对应目标右手；标签保留为配置项，避免不同镜像链路下写死。

### MANO/SMPL-X 后端

`tracking/mano_adapter.py` 接受完整 MANO 45 维 pose 和 betas，使用用户合法获取的
MANO assets 前向生成 21 点。实时 HaMeR/Dyn-HaMR 后端应同时输出：

- 右手 21 点或 MANO pose；
- 检测置信度；
- D435 深度反投影的腕部位置；
- 时间戳。

MANO assets、模型 checkpoint 和第三方代码不提交到本项目。

## 5. 标定

### D435 到 UR5 方向

初始相对零点只能消除平移偏置，不能确定轴方向。把机器人放在工作区中央，在
dry-run 下分别做右、上、前后三种单轴动作，修正配置中的 3x3 旋转矩阵。

若相机安装姿态不是轴对齐，应通过相机外参/手眼标定得到完整旋转矩阵，而不是
使用示例中的轴置换矩阵。

### 人手特征

按 `o` 记录自然张开，按 `f` 记录自然握拳。每个操作者应独立记录，以消除手型和
活动范围差异。

### O6

配置里的位置全部是占位值。实机采用最低可用力矩、低速，每次只移动一个轴，
记录张开、闭合、死区及不会发生自碰撞的安全范围。必须根据实际运动确认位置 4、
5 分别对应无名指和小指，不能只依赖 GUI 标签。

## 6. 安全状态机

```text
WAITING_CALIBRATION --c--> DISARMED --a--> ARMED
                              ^             │
                              └────a────────┘
                                            │ tracking timeout
                                            v
                                      TRACKING_LOST

any state --e/error--> ESTOP (restart required)
```

硬件模式额外要求命令行风险确认，不允许自动使能。三个独立限制依次生效：相对动作
范围、绝对 TCP 工作空间、单周期最大 TCP 步长。跟踪丢失时调用 `servoStop()`，不会
依据最后速度继续外推。

## 7. 阶段 B 扩展

阶段 B 不应简单继续复制人手闭合量。建议状态切换为：

```text
PREGRASP_MIMIC -> FIRST_CONTACT -> FORCE_LIMITED_CLOSE -> HOLD -> RELEASE
```

接触前使用本 Demo 的姿态模仿；首次接触后，以 O6 压力、UR5 TCP force 和本项目
damage/slip outcome model 覆盖人的继续闭合命令。只有确认 O6 传感器型号、空载
基线、采样顺序和饱和值后，才能启用当前预留的 `ForceGuard`。

数据建议保持分离：

```text
ur5_action:       [T, 6]
o6_command:       [T, 6]
o6_state:         [T, 6]
o6_force:         [T, ...]
human_wrist:      [T, 3]
human_hand_joint: [T, 21, 3]
```

因此 UR5 六维动作与 O6 六维动作合并时是 12 维，而不是当前平行夹爪路线的 7 维。
