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

## HMWM Stage 1 诊断

在修改模型前，对同一 checkpoint 同时诊断 train/val 的逐 horizon wrist error、
逐 clip error、CV correction 和三组状态尺度：

```bash
python -m smplx_hand_lwm.diagnose_hmwm \
  --checkpoint /data/chy/hot3d_hand_lwm/pilot50/stage1_checkpoints/best.pt \
  --train-manifest /data/chy/hot3d_hand_lwm/pilot50/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --batch-size 64 \
  --num-workers 0 \
  --output /data/chy/hot3d_hand_lwm/pilot50/hmwm_v0_diagnostics.json
```

该步骤不修改或训练模型。根据 train/val generalization gap、逐时域误差和
predicted/target CV correction ratio，再选择下一项单变量实验。

审计 teacher LA 是否包含未来 wrist/MANO 信息：

```bash
python -m smplx_hand_lwm.audit_teacher_la \
  --checkpoint /data/chy/hot3d_hand_lwm/pilot50/stage1_checkpoints/best.pt \
  --train-manifest /data/chy/hot3d_hand_lwm/pilot50/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --batch-size 64 \
  --num-workers 0 \
  --ridge-alpha 1.0 \
  --output /data/chy/hot3d_hand_lwm/pilot50/teacher_la_audit.json
```

线性 probe 只在 train teacher-LA 上拟合。Train/val R2 都高说明对应运动量在
LA 中可跨 clip 读取；train 高而 val 低说明 latent semantics 只在训练 clips
内成立。该审计不会更新 Hand-IDM 或 HMWM 参数。

## IDM-A1：wrist-aware teacher LA

IDM-A1 只在 Hand-IDM 的 `latent_mean` 后增加训练期线性辅助头，预测未来
12 帧相对于 constant-velocity 外推轨迹的 `12 x 3` wrist correction。
HMWM 的结构、CV anchor 和推理接口均保持不变。辅助损失权重为 5，并使用
独立目录保存结果，不覆盖 pilot50 基线：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_pilot50_idm_a1.yaml
```

训练完成后，对新 checkpoint 重复完全相同的 LA audit：

```bash
python -m smplx_hand_lwm.audit_teacher_la \
  --checkpoint /data/chy/hot3d_hand_lwm/pilot50_idm_a1/stage1_checkpoints/best.pt \
  --train-manifest /data/chy/hot3d_hand_lwm/pilot50/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --batch-size 64 \
  --num-workers 0 \
  --ridge-alpha 1.0 \
  --output /data/chy/hot3d_hand_lwm/pilot50_idm_a1/teacher_la_audit.json
```

本轮首要判据是 `wrist_cv_correction` 的 validation R2 从 -0.3589 提升到
大于 0，同时 probe validation trajectory ADE 低于 32.2 mm。还需确认
rotation 与 MANO probe 没有明显退化，再评估 HMWM wrist ADE/FDE。

## IDM-A2：window-local wrist translation

IDM-A2 保留 A1 auxiliary head 和所有 A1 超参数，只在状态进入 Hand-IDM
之前，将 context/future 的 wrist translation 同时减去最后一帧 context 的
wrist translation。HMWM 仍接收原始坐标状态，结构和输出坐标均不改变：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_pilot50_idm_a2.yaml
```

训练完成后采用与 A1 相同的 probe 协议：

```bash
python -m smplx_hand_lwm.audit_teacher_la \
  --checkpoint /data/chy/hot3d_hand_lwm/pilot50_idm_a2/stage1_checkpoints/best.pt \
  --train-manifest /data/chy/hot3d_hand_lwm/pilot50/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --batch-size 64 \
  --num-workers 0 \
  --ridge-alpha 1.0 \
  --output /data/chy/hot3d_hand_lwm/pilot50_idm_a2/teacher_la_audit.json

python -m smplx_hand_lwm.evaluate_stage1 \
  --checkpoint /data/chy/hot3d_hand_lwm/pilot50_idm_a2/stage1_checkpoints/best.pt \
  --manifest /data/chy/hot3d_hand_lwm/pilot50/val.jsonl \
  --batch-size 64 \
  --num-workers 0 \
  --output /data/chy/hot3d_hand_lwm/pilot50_idm_a2/stage1_val_metrics.json
```

## Model-S1：扩大 Hand-IDM 与 HMWM

Model-S1 是 A2 的容量对照实验，只把 shared hidden dimension 从 256 提高到
512、encoder/decoder 从 4 层提高到 8 层、FFN 从 1024 提高到 2048、
attention heads 从 8 提高到 16。LA 维度仍为 64，其余数据、损失和训练参数
不变。总参数量约从 7.48M 增加到 59.06M：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_pilot50_idm_a2_model_s1.yaml
```

该实验用于区分 capacity bottleneck 与 data/generalization bottleneck。如果
train 显著改善而 val 不改善，则不应继续盲目扩容，应先增加独立 clips。

## Data-D1：200 clips，固定 pilot50 validation

Data-D1 保持59.06M参数的 Model-S1完全不变，将 HOT3D Quest3 clips 从50个
扩展到200个。原 pilot50 的10个 validation clips 固定不动，其余有效 clips
全部用于训练，目标为190 train clips / 10 validation clips：

```bash
python -m smplx_hand_lwm.scripts.prepare_hot3d \
  --clips-dir /data/chy/hot3d_clips/train_quest3 \
  --output-dir /data/chy/hot3d_hand_lwm/data_d1_200/tracks \
  --train-manifest /data/chy/hot3d_hand_lwm/data_d1_200/train.jsonl \
  --val-manifest /data/chy/hot3d_hand_lwm/data_d1_200/fixed_val10.jsonl \
  --val-clip-ids 000004,000009,000011,000016,000019,000023,000025,000026,000029,000045 \
  --handedness right \
  --max-clips 200

python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_model_s1.yaml
```

固定 validation 是为了让 Data-D1 与 pilot50 Model-S1 的验证指标可以直接
比较。若某个固定 validation clip 转换失败，准备脚本会直接报错，不会静默
改变评估集合。

## Test-D1：完全独立的50 clips

Test-D1 使用 `000200-000249`，不重新训练 Data-D1，也不参与 checkpoint
选择。test-only 准备脚本要求传入现有 train/val manifests，并在发现任何
clip overlap 时终止：

```bash
python -m smplx_hand_lwm.scripts.prepare_hot3d_test \
  --clips-dir /data/chy/hot3d_clips/train_quest3 \
  --output-dir /data/chy/hot3d_hand_lwm/test_d1_50/tracks \
  --test-manifest /data/chy/hot3d_hand_lwm/test_d1_50/test.jsonl \
  --clip-start 200 \
  --clip-end 250 \
  --target-valid-clips 50 \
  --exclude-manifest /data/chy/hot3d_hand_lwm/data_d1_200/train.jsonl \
  --exclude-manifest /data/chy/hot3d_hand_lwm/data_d1_200/fixed_val10.jsonl \
  --handedness right
```

候选范围比目标多一个 clip。选择规则固定为按 ID 排序后，取前50个能够产生
至少一个完整16帧窗口的 clips；这允许跳过纯粹由可见性缺口导致的空 clip，
但不会根据模型结果选择测试样本。

随后直接使用 Data-D1 的 epoch-74 `best.pt` 运行 LA audit、Stage-1 evaluation
和 HMWM diagnostics，禁止用 Test-D1 重新选择 checkpoint。

## HMWM-LaWM-v0：AdaLN-Zero decoder

完成固定 validation 和完全独立 Test-D1 后，下一项单变量实验只替换 HMWM
decoder。Hand-IDM、64D Gaussian/VAE LA、Model-S1 容量、Data-D1 划分、
A1/A2、损失权重和训练参数全部不变。

`HMWM-LaWM-v0` 将当前的 learned future queries、单次 LA 加法和
TransformerDecoder cross-attention 替换为 LaWM-style decoder：

```text
constant-velocity future anchor + fixed 1D horizon position
  -> AdaLN-Zero self-attention blocks, each conditioned on teacher LA
  -> residual future hand trajectory
```

训练前运行双 decoder smoke test：

```bash
python -m smplx_hand_lwm.scripts.smoke_test_stage1
```

正式训练：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_hmwm_lawam_v0.yaml
```

训练结束后，必须在 fixed validation 和既有 Test-D1 上分别运行
`evaluate_stage1`、`audit_teacher_la` 和 `diagnose_hmwm`。Test-D1 只用于最终
比较，禁止据此重新选择 checkpoint 或修改超参数。

全部评估可用单个入口执行。脚本会生成7个独立 JSON、对应日志，以及汇总文件
`evaluation_bundle.json`：

```bash
python -m smplx_hand_lwm.scripts.run_hmwm_lawam_v0_evaluation
```

如果评估中断，可复用已经生成且能正常解析的 JSON：

```bash
python -m smplx_hand_lwm.scripts.run_hmwm_lawam_v0_evaluation \
  --skip-existing
```

## HMWM-LaWM-v1：加入完整 context tokens

v0 在独立 Test-D1 上证明 AdaLN 会使用 LA，但丢失完整 context 后出现明显的
train/test gap。v1 保留 v0 的 AdaLN-Zero、CV anchor、64D LA、模型容量、数据、
损失与优化设置，只在 self-attention 输入前拼接4个 context hand-state tokens：

```text
[4 context tokens at positions -4..-1;
 12 CV-anchor future tokens at positions 0..11]
  -> shared AdaLN-Zero blocks conditioned on teacher LA
  -> select final 12 tokens
  -> residual future hand trajectory
```

固定位置编码不含可训练参数，因此 v0 与 v1 参数量相同。训练命令：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_hmwm_lawam_v1_context.yaml
```

## HMWM-AdaLN-Cross-v2：恢复 read-only context memory

v0/v1 的 self-attention tokenization 在独立 Test-D1 上均不如原 HMWM。v2
恢复原 Model-S1 的 learned future queries 和完整 context cross-attention，只
替换 LA conditioning：移除输入 queries 上的一次 LA 加法，改为每层六路
AdaLN-Zero。context tokens 始终只作为 cross-attention K/V，不接受 LA 调制，
也不在 decoder 中被更新：

```text
learned future queries
  -> AdaLN-Zero self-attention (teacher LA)
  -> cross-attention(read-only 4-frame context memory)
  -> AdaLN-Zero FFN (teacher LA)
  -> repeated for every decoder block
```

正式训练：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_hmwm_adaln_cross_v2.yaml
```

## Frozen-v2-IDM decoder isolation

为区分 v2 的收益来自 Hand-IDM 还是 decoder，固定使用 v2 `best.pt` 中的
Hand-IDM，并强制 `eval + no_grad + posterior mean`。两组实验均从头训练 decoder：

```bash
python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_frozen_v2_idm_original_decoder.yaml

python -m smplx_hand_lwm.train_stage1 \
  --config smplx_hand_lwm/configs/stage1_hot3d_data_d1_200_frozen_v2_idm_adaln_cross.yaml
```

训练完成后使用一个入口完成 fixed-val/Test-D1 全套评估、验证两个 checkpoint
中的 IDM 参数哈希一致，并生成逐指标差值：

```bash
python -m smplx_hand_lwm.scripts.run_frozen_v2_idm_decoder_comparison
```

主结论以 `decoder_comparison.json` 的 Test-D1 为准；其中
`original_minus_adaln < 0` 表示原 TransformerDecoder 更好。Test-D1 不参与
checkpoint 选择或调参。

数据格式见 `datasets/schema.md`，模型设计和实验计划分别见 `docs/model_architecture.md` 与 `docs/experiment_plan.md`。
