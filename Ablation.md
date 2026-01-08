# 消融实验
## A组：先定义一个“最强基线” Baseline（B0）

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

已完成，就是我们目前的提交的模型，验证集 84.65，测试集 55.3

✅ Best threshold (maximize accuracy) = 0.3900, accuracy = 55.30%

预测结果分布:
  - Pred 0 (Real): 491 (49.10%)
  - Pred 1 (Fake): 509 (50.90%)

概率统计:
  - mean   : 0.4480
  - median : 0.3981
  - min/max: 0.0189 / 0.9903
  - std    : 0.2901

✅ Accuracy @thr=0.3900: 55.30% (n=1000)

=== 模型评测报告 ===
对齐样本数: 1000

— 混淆矩阵（正类=Fake=1） —
TP: 281  FP: 228
FN: 219  TN: 272

— 全局指标 —
Accuracy: 0.5530
Balanced Accuracy: 0.5530
ROC-AUC: 0.5542

— 分类别指标 —
Real(0):
  Precision: 0.5540  Recall: 0.5440  F1: 0.5489  正确率: 0.5440
Fake(1):
  Precision: 0.5521  Recall: 0.5620  F1: 0.5570  正确率: 0.5620

— 支持度（样本数） —
Real(0): 500   Fake(1): 500   Total: 1000

— 汇总 —
Macro F1: 0.5530   Weighted F1: 0.5530

## B组：结构优化
### 去掉 Global stream，只用 Local
证明“Global stream 真的有用吗？

只用 Local（关掉 global-local）
```bash
# 只改这一项：去掉 --use_global_local
python train.py \
  --use_learnable_srm --fusion_mode concat --patch_topk 5 --batch_size 16 --num_epochs 50
```

模型已经训练完成
验证集 61.60 
测试集 53.10


预测结果分布:
  - Pred 0 (Real): 719 (71.90%)
  - Pred 1 (Fake): 281 (28.10%)

概率统计:
  - mean   : 0.4668
  - median : 0.4659
  - min/max: 0.3676 / 0.7329
  - std    : 0.0583

✅ Accuracy @thr=0.5000: 53.10% (n=1000)

### 证明“噪声分支（SRM）与可学习性”的价值
SSP 的核心在于噪声。这组实验将验证：1. SRM 分支是否必须？ 2. 让它“可学习”是否有额外收益？

固定 SRM（非可学习版）
验证固定先验是否足够，或者梯度微调是否能带来更好的泛化。

```bash
# 只改这一项：去掉 --use_learnable_srm（代码中该参数控制 requires_grad）
python src/train.py \
  --use_global_local --fusion_mode concat --feature_fusion_type concat \
  --global_size 384 --patch_size 32 --patch_topk 5 \
  --jpeg_p_global 0.3 --jpeg_p_patch 0.1 --blur_p 0.15 \
  --batch_size 32 --num_epochs 50 


# 不加 --use_learnable_srm，默认不启用
```

验证集 83.65 
测试集 47.60

预测结果分布:
  - Pred 0 (Real): 582 (58.20%)
  - Pred 1 (Fake): 418 (41.80%)

概率统计:
  - mean   : 0.4731
  - median : 0.4396
  - min/max: 0.0478 / 0.9714
  - std    : 0.2118

✅ Accuracy @thr=0.5000: 47.60% (n=1000)

## C组：证明“Patch 选择策略”的影响
这组实验验证选取“简单区域”以及“Top-K”机制的必要性。

关掉 Top-K 投票（只取 Top-1）
验证多 Patch 联合判定是否比单一 Patch 更稳健。

```bash
# 只改这一项：--patch_topk 1
python src/train.py \
  --use_global_local --use_learnable_srm --fusion_mode concat \
  --patch_topk 1 \
  --global_size 384 --patch_size 32 --batch_size 16 --num_epochs 50

```

验证集 84.20
测试集 50.30

预测结果分布:
  - Pred 0 (Real): 643 (64.30%)
  - Pred 1 (Fake): 357 (35.70%)

概率统计:
  - mean   : 0.4457
  - median : 0.4145
  - min/max: 0.0356 / 0.9253
  - std    : 0.1850

✅ Accuracy @thr=0.5000: 50.30% (n=1000)

## D组：证明“数据增强（跨域泛化技术）”的效果
### 无 FDA 频域增强

```bash
# 需要在 transform.py 中关闭 FDA 或将概率设为 0（如果 train.py 有对应参数）
# 假设参数名为 --fda_p 0
python src/train.py \
  --use_global_local --use_learnable_srm --fda_p 0 \
  --global_size 384 --patch_size 32 --batch_size 16 --num_epochs 50
```

就是我们之前训练的那个版本
验证集 82.45，测试集 52.1

=== 模型评测报告 ===        
对齐样本数: 1000

— 混淆矩阵（正类=Fake=1） — 
TP: 225  FP: 204
FN: 275  TN: 296

— 预测分布 —
Pred.Real: 57.10% (571/1000)
Pred.Fake: 42.90% (429/1000)

— 全局指标 —
Accuracy: 0.5210
Balanced Accuracy: 0.5210   
ROC-AUC: 0.5195

— 预测概率统计 —
全局: μ=0.4563, σ=0.3249
Real(0): μ=0.4467, σ=0.3262
Fake(1): μ=0.4659, σ=0.3232

— 分类别指标 —
Real(0):
  Precision: 0.5184  Recall: 0.5920  F1: 0.5528  正确率: 0.5920
Fake(1):
  Precision: 0.5245  Recall: 0.4500  F1: 0.4844  正确率: 0.4500

— 支持度（样本数） —
Real(0): 500   Fake(1): 500   Total: 1000

— 汇总 —
Macro F1: 0.5186   Weighted F1: 0.5186



### 去掉所有退化增强 (Baseline - No Degradation)
验证 JPEG 压缩、模糊等增强是否能提升对低质量图的识别。

```bash
# 将所有增强概率设为 0
python src/train.py \
  --use_global_local --use_learnable_srm \
  --jpeg_p_global 0 --jpeg_p_patch 0 --blur_p 0 --resample_p 0 --noise_p 0 --freq_p 0 \
  --batch_size 16 --num_epochs 50

```
验证集 84.55
测试集 46.80

预测结果分布:
  - Pred 0 (Real): 698 (69.80%)
  - Pred 1 (Fake): 302 (30.20%)

概率统计:
  - mean   : 0.3839
  - median : 0.3358
  - min/max: 0.0097 / 0.9600
  - std    : 0.2300

✅ Accuracy @thr=0.5000: 46.80% (n=1000)

