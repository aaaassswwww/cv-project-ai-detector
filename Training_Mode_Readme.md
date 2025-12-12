# 训练参数文档
## 训练参数
### 🟦 **A. 基础训练参数**

| 参数                | 默认   | 含义         | 是否影响性能      |
| ----------------- | ---- | ---------- | ----------- |
| `--num_epochs`    | 50   | 总训练轮数      | ⭐ 强影响       |
| `--batch_size`    | 32   | batch 大小   | 中等影响        |
| `--learning_rate` | 1e-4 | 初始学习率      | ⭐ 强影响       |
| `--weight_decay`  | 1e-4 | AdamW 的正则项 | 中等影响        |
| `--seed`          | 42   | 随机种子       | 不影响性能（影响复现） |

---

### 🟦 **B. Patch 采样相关参数（影响 local stream 核心特征）**

| 参数                   | 默认  | 说明                 | 对性能影响               |
| -------------------- | --- | ------------------ | ------------------- |
| `--patch_size`       | 32  | patch 的尺寸（从原图裁）    | ⭐ 强影响（太小会损失结构）      |
| `--patch_topk`       | 3   | 每张图选 K 个最重要的 patch | ⭐ 强影响（K 越大越稳，但计算更大） |
| `--patch_var_thresh` | 5.0 | 去掉平坦 patch         | 中等（大多数情况有益）         |

---

### 🟦 **C. 强增强（patch-level + global-level）**

你现在的 transform 中包含丰富的数据增强：
JPEG、模糊、降采样、高频扰动、噪声……

| 参数                       | 默认    | 功能                        | 建议             |
| ------------------------ | ----- | ------------------------- | -------------- |
| `--jpeg_p_global`        | 0.2   | 全局 JPEG（用于 global stream） | ⭐ 必须开          |
| `--jpeg_p_patch`         | 0.05  | patch JPEG                | ⭐ 强烈推荐开（提升鲁棒性） |
| `--jpeg_quality_min/max` | 30–95 | JPEG 质量范围                 | 默认合理           |
| `--blur_p`               | 0.15  | 模糊                        | 保持             |
| `--resample_p`           | 0.15  | 下采样重采样                    | ⭐ 对检测生成图像有效    |
| `--noise_p`              | 0.1   | 加噪声                       | 可开可不开          |
| `--freq_p`               | 0.1   | 高频扰动                      | ⭐ 对鲁棒性帮助大      |
| `--freq_radius`          | 0.25  | 高频扰动范围                    | 默认合理           |

**这些是你模型强度的很大来源，不建议删除。**

---

### 🟦 **D. Learnable SRM 模块（Local branch 的增强版）**

| 参数                    | 默认      | 含义                    | 建议                 |
| --------------------- | ------- | --------------------- | ------------------ |
| `--use_learnable_srm` | False   | 是否启用可学习 SRM           | ⭐ 强烈建议开启           |
| `--srm_out_channels`  | 12      | 输出通道数                 | 12 足够              |
| `--srm_kernel_size`   | 5       | kernel 大小             | 5 更强               |
| `--srm_use_norm`      | False   | GroupNorm             | ⭐ 建议打开             |
| `--fusion_mode`       | replace | SRM 如何与 RGB 融合        | ⭐ 建议用 concat（性能最好） |
| `--srm_freeze_epochs` | 0       | 前几轮冻结 SRM             | 视情况                |
| `--srm_use_mixing`    | False   | 1x1 卷积 channel mixing | ⭐ 建议开启             |
| `--srm_lr_scale`      | 1.0     | SRM lr 缩放             | OK                 |

---

### 🟦 **E. Global-local 双流架构**

| 参数                      | 默认     | 含义                                | 建议                |
| ----------------------- | ------ | --------------------------------- | ----------------- |
| `--use_global_local`    | False  | 是否启用 Global-Local Dual Stream     | ⭐⭐ 性能飞跃，强烈推荐      |
| `--global_size`         | 384    | global 分支的输入尺寸                    | ⭐ 建议设为 384 或 512  |
| `--share_backbone`      | False  | local/global 是否共享 ResNet backbone | 性能更高 = False（不共享） |
| `--feature_fusion_type` | concat | global/local 特征融合方式               | ⭐ concat（最佳）      |

---

### 🟦 **F. 学习率调度与 Warmup / Early stop**

| 参数                      | 默认   | 含义             |
| ----------------------- | ---- | -------------- |
| `--warmup_epochs`       | 3    | warmup 轮数      |
| `--early_stop_patience` | 8    | 早停 patience    |
| `--min_delta`           | 1e-4 | early stop 稳定度 |
| `--eta_min`             | 1e-6 | cos退火最低学习率     |


## 实验/消融
## 先定义一个“最强基线” Baseline（B0）

后面所有消融都基于它“只改一两项”：

```bash
python src/train.py \
  --use_global_local \
  --use_learnable_srm \
  --fusion_mode concat \
  --feature_fusion_type concat \
  --global_size 384 \
  --patch_size 32 \
  --patch_topk 5 \
  --patch_var_thresh 5.0 \
  --srm_out_channels 12 \
  --srm_kernel_size 5 \
  --srm_use_norm \
  --srm_use_mixing \
  --jpeg_p_global 0.3 \
  --jpeg_p_patch 0.1 \
  --blur_p 0.15 \
  --resample_p 0.15 \
  --noise_p 0.1 \
  --freq_p 0.1 \
  --freq_radius 0.25 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --num_epochs 50
```

---

## A 组：证明“Global stream 真的有用吗？”（最高优先级）

### A1. 只用 Local（关掉 global-local）

```bash
# 只改这一项：去掉 --use_global_local
python train.py \
  --use_learnable_srm --fusion_mode concat --patch_topk 5 --batch_size 16 --num_epochs 50
```

对比 B0 vs A1：能回答“global 带来的增益到底有多大”。

### A2. Global 变小/变大（global_size 消融）

```bash
# global_size=320
... --global_size 320

# global_size=512
... --global_size 512
```

用途：确定 “384 是否最佳点”，也能观察 overfit 风险。

---

## B 组：证明“SRM/learnable SRM 是否必要？”

### B1. 关掉 learnable SRM（退回 classic SRM）

```bash
# 去掉 --use_learnable_srm（其他不变）
... --use_global_local --fusion_mode concat --feature_fusion_type concat ...
```

### B2. Learnable SRM 但不做 mixing / norm（组件级消融）

```bash
# 去掉 mixing
... --srm_use_norm  (不加 --srm_use_mixing)

# 去掉 norm
... --srm_use_mixing (不加 --srm_use_norm)

# 两者都去掉
... (不加 --srm_use_norm 也不加 --srm_use_mixing)
```

用途：论文里最好写的 ablation（告诉读者哪个模块贡献最大）。

### B3. SRM kernel / channel 容量

```bash
# kernel=3
... --srm_kernel_size 3

# out_channels=16（更大容量）
... --srm_out_channels 16
```

---

## C 组：融合策略消融（证明你选 concat 合理）

### C1. Local fusion_mode：replace vs concat vs dual_stream

```bash
# replace
... --fusion_mode replace
# concat
... --fusion_mode concat
# dual_stream
... --fusion_mode dual_stream
```

### C2. Feature fusion：concat vs add vs attention

```bash
# add（轻量）
... --feature_fusion_type add

# attention（高性能但更敏感）
... --feature_fusion_type attention
```

用途：证明“融合怎么做最好”，并且 attention 如果没提升，也能合理解释“更复杂不一定更强”。

---

## D 组：Patch 相关消融（决定性能/算力最关键的超参）

### D1. topk：1 / 3 / 5 / 7

```bash
... --patch_topk 1
... --patch_topk 3
... --patch_topk 5
... --patch_topk 7
```

用途：得到“性能-算力曲线”，通常论文里非常有说服力。

### D2. patch_size：16 / 32 / 64（局部纹理 vs 结构信息）

```bash
... --patch_size 16
... --patch_size 32
... --patch_size 64
```

### D3. patch_var_thresh：0 vs 5（过滤平坦 patch 是否有用）

```bash
... --patch_var_thresh 0
... --patch_var_thresh 5.0
```

---

## E 组：增强消融（证明鲁棒性来自你的增强设计）

这组建议你**只做“开/关”对照**，最直观。

### E1. 关掉全局 JPEG（证明 global JPEG 的价值）

```bash
... --jpeg_p_global 0
```

### E2. 关掉 patch JPEG

```bash
... --jpeg_p_patch 0
```

### E3. 只保留 JPEG，其他增强都关（看增强到底是否必要）

```bash
... --blur_p 0 --resample_p 0 --noise_p 0 --freq_p 0
```

### E4. 只关掉 freq（高频扰动）看看是否贡献最大

```bash
... --freq_p 0
```

---

# F 组：训练超参敏感性（可选，但很实用）

### F1. 学习率 3 点

```bash
... --learning_rate 5e-5
... --learning_rate 1e-4
... --learning_rate 2e-4
```

### F2. label_smoothing：0 vs 0.1（你现在默认 0.1）

```bash
... --label_smoothing 0
... --label_smoothing 0.1
```

有些检测任务 label smoothing 会“拉低表观 acc 但提高泛化”，这很值得写进 ablation。

---

## 我建议你怎么跑（省时间的顺序）

1. **A1 → A2**：先证明 global stream 价值 + 确定 global_size
2. **B1/B2**：证明 learnable SRM & 组件贡献
3. **C1/C2**：融合策略
4. **D1**：topk 曲线（论文非常加分）
5. **E1/E2/E3**：增强鲁棒性来源
6. 最后再做 F 组（如果你想 squeeze 最后 0.5~1%）
