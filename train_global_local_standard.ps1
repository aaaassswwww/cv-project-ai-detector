# Global-Local Dual Stream 快速训练脚本
# 根据不同需求选择合适的配置

# ============================================
# 配置 1: Standard（推荐，平衡性能和速度）
# ============================================
python src/train.py `
    --use_global_local `
    --use_learnable_srm `
    --srm_out_channels 12 `
    --srm_use_norm `
    --srm_lr_scale 0.1 `
    --fusion_mode concat `
    --global_size 384 `
    --feature_fusion_type concat `
    --num_epochs 50 `
    --batch_size 32 `
    --learning_rate 1e-4 `
    --weight_decay 1e-4 `
    --warmup_epochs 3 `
    --early_stop_patience 8 `
    --patch_topk 1 `
    --model_name global_local_standard `
    --seed 42

# 预期性能:
# - 准确率: ~92-93%
# - 显存: ~7GB
# - 速度: ~1.5s/batch
# - 参数量: 49.3M

Write-Host "Standard configuration training started..." -ForegroundColor Green
Write-Host "Expected accuracy: ~92-93%" -ForegroundColor Cyan
Write-Host "VRAM usage: ~7GB" -ForegroundColor Cyan
