# 实验计划：SMPL-X Hand Latent World Model

## 1. 核心问题

实验不只验证 hand-state regression 是否收敛，而要回答四个问题：

1. 用结构化手部状态替代视觉 latent，能否准确预测未来手部运动？
2. latent action 是否表示动作变化，而不是复制未来姿态？
3. 预测轨迹经过重定向后，是否仍保持合理的指尖接触关系？
4. 该动作先验能否提高易损物体候选抓取的成功率并降低损伤率？

## 2. 分阶段实验

### E0：数据恢复质量审计

先选 50–100 条短 ego 片段，不训练模型，比较至少两种手部恢复后端。人工标注一小部分 2D keypoints/contact frame，检查：

- 2D reprojection error；
- 相邻帧 joint jitter；
- 左右手 identity switch rate；
- 抓取接触阶段的有效帧比例；
- 遮挡与快速运动条件下的失败率。

通过门槛后才能扩大离线提取。严重遮挡窗口必须过滤或带 `valid mask`，不能作为无噪声监督。

### E1：小规模 pipeline 验证

目标：验证 schema、切窗、前向、反向和 checkpoint。

- 1k–10k windows；
- 4 帧 context，12 帧 future，30 FPS；
- 单手 canonical track；
- 在 128 个窗口上做 overfit test；
- 检查 latent KL、轨迹误差和 contact loss 是否同时下降。

成功标准：训练集可稳定过拟合，验证输出无 NaN，预测序列维度与关节约束正确。该阶段不能声称具备真实机器人泛化能力。

HOT3D pilot 首先使用官方 15 维 MANO PCA pose，加上腕部 3D 平移和 6D
旋转形成 24 维状态。先用 3 个 clips 验证转换和训练，再扩展至 50 clips。
MANO assets 未接入时只报告参数空间与腕部轨迹误差，不报告 MPJPE。

### E2：未来手部轨迹预测

按人物、场景和物体实例分组划分，禁止同一原视频相邻窗口跨 train/val/test 泄漏。

建议数据规模：

```text
pilot: 10k–50k windows
main:  200k+ windows
```

主要指标：

- MPJPE / fingertip MPJPE，毫米；
- wrist ADE/FDE，毫米；
- rotation error，度；
- contact precision/recall/F1；
- velocity/acceleration error；
- 1、5、10 个采样候选的 best-of-K error。

报告短、中、长三个 horizon，例如 0.2 s、0.4 s、1.0 s。

### E3：latent action 质量

验证 `z_h` 是否包含可迁移的动作语义：

- future retrieval：给定 `H_t,z_h` 检索正确 `H_T`；
- action classification linear probe；
- 同一动作跨人物/场景的聚类；
- latent swapping：将片段 A 的 `z_h` 与片段 B 的 `H_t` 组合；
- posterior collapse：监测 active latent dimensions、KL 和 decoder 对 `z_h` 的敏感度。

必须加入 `shuffle-z` 测试：随机打乱 batch 内的 latent action。如果预测几乎不变，说明 decoder 忽略了 `z_h`。

在 Model-S1、Data-D1 和独立 Test-D1 固化后，运行 `HMWM-LaWM-v0` 单变量
实验：只把原 Transformer decoder 换成逐层 latent-action-conditioned
AdaLN-Zero decoder。比较 fixed validation 与 Test-D1 上的 posterior、
shuffle-z、constant velocity、per-horizon error 和 teacher-LA probes；禁止用
Test-D1 选择 checkpoint。只有训练重建改善而 Test-D1 不改善时，应判定为
更强 decoder 利用了 clip-specific LA，不能据此继续增加 HMWM 容量。

若 v0 呈现训练重建增强而独立 Test-D1 退化，则运行 `HMWM-LaWM-v1`：保持
v0 全部设置，仅把完整 context hand tokens 拼接到 future-anchor tokens 前。
该实验用于隔离 v0 的退化是否来自 decoder 丢失完整 context history，而不是
继续调整 AdaLN 容量或损失权重。

若 v1 仍不如原 HMWM，则运行 `HMWM-AdaLN-Cross-v2`。该实验恢复原
TransformerDecoder 的 learned future query/read-only context memory 关系，
只将 one-shot LA addition 替换为逐层六路 AdaLN-Zero。它用于隔离失败来源
究竟是 LaWM-style tokenization，还是 AdaLN conditioning 本身。

若 v2 的 teacher LA probe 改善但 decoder 指标仍未超过原 HMWM，则冻结 v2
Hand-IDM，使用确定性 posterior mean 为每个窗口生成固定 teacher LA，分别从头
训练原 TransformerDecoder 与 AdaLN-Cross decoder。两组仅允许 decoder 类型和
checkpoint 输出目录不同。最终比较前必须验证两个 checkpoint 的
`inverse_dynamics.*` 参数哈希一致，主判据为完全独立 Test-D1 的 posterior
wrist ADE/FDE、rotation6D MAE、MANO PCA MAE，并同时报告 shuffle-z gap。

### E3.1：current-only latent prior

冻结通过 E2 的 IDM 与 Hand-LWM。IDM 以真实未来产生 teacher latent，prior
仅输入 `H_context`，通过 latent distillation 与 frozen Hand-LWM rollout loss
联合训练。主结果必须使用 prior rollout，而不是 future-conditioned posterior：

- current-only prior vs Last Pose；
- current-only prior vs Constant Velocity；
- posterior teacher 仅作为重建上界；
- latent mean MAE；
- 单一 prior mean 与 best-of-K prior samples。

### E4：机器人重定向离线评估

先在仿真或 URDF kinematics 中进行，不直接上真实易损物体：

- MANO fingertip 到 Linker Hand O6 fingertip error；
- joint-limit violation；
- self-collision / object penetration rate；
- trajectory jerk；
- IK success rate；
- contact-order preservation。

对比逐关节映射、优化式 fingertip retargeting 和学习式 retargeting。

### E5：UR5 + 灵巧手真实闭环评估

在 O6 硬件就绪后开展。Hand-LWM 只生成候选动作；当前项目的 tactile-force outcome model 负责筛选。

任务从海绵开始，再扩展至果冻、豆腐和软糕点。逐级降低刚度，设置明确的急停、力阈值和最小速度。

最终指标：

- grasp success rate；
- real damage rate；
- slip rate；
- peak force / force overshoot；
- unseen object generalization；
- inference latency；
- 相对固定参数和无 outcome filtering 的提升。

## 3. Baselines

### 预测基线

1. Last pose / constant velocity；
2. GRU/LSTM future predictor；
3. Transformer deterministic predictor；
4. CVAE hand predictor；
5. 本方法 Hand-IDM + Hand-LWM；
6. 可选视觉 latent LaWAM-style teacher，作为信息量上界对照。

### 机器人基线

1. 固定灵巧手预抓姿态；
2. 最近邻人手轨迹重定向；
3. deterministic hand predictor + retargeting；
4. Hand-LWM best-of-K + retargeting；
5. Hand-LWM best-of-K + tactile-force outcome filtering。

## 4. 必做消融

- 单帧 `H_t` vs 多帧 context；
- camera-relative vs stabilized-world vs object-relative coordinates；
- axis-angle vs rotation-6D；
- 只预测终点 vs 预测完整轨迹；
- 无 KL / 不同 latent dimension；
- absolute prediction vs residual prediction；
- 无 joint auxiliary head；
- 无 contact head；
- 无 velocity/acceleration loss；
- 单一预测 vs best-of-K 多模态预测。

## 5. 数据划分与统计要求

- 主结果至少使用 3 个随机种子；
- 按 subject、object instance、source video 分组划分；
- 报告均值、标准差和置信区间；
- 同时报告 hand reconstruction 有效率，避免只在容易样本上评测；
- 自动恢复的 SMPL-X/MANO 是 pseudo-label，论文中必须明确说明。

## 6. Go / No-Go 决策

进入重定向实验前，至少满足：

- 明显优于 constant-velocity baseline；
- fingertip 误差随 horizon 增长保持可控；
- contact F1 有实际区分能力；
- shuffle-z 后性能显著下降；
- unseen subject/object 上没有灾难性退化。

进入真实易损物体实验前，至少满足：

- 仿真 IK success rate 和 joint-limit compliance 达标；
- 碰撞/穿透检查通过；
- 有独立的力阈值安全层；
- tactile-force outcome model 已在真实机器人数据上验证，而非只有 dummy pipeline。
