# Global-Local Dual Stream - High Performance 配置
# 追求最高精度

python src/train.py `
    --use_global_local `
    --use_learnable_srm `
    --srm_out_channels 16 `
    --srm_use_norm `
    --srm_use_mixing `
    --srm_lr_scale 0.1 `
    --fusion_mode dual_stream `
    --global_size 512 `
    --feature_fusion_type attention `
    --num_epochs 50 `
    --batch_size 16 `
    --learning_rate 1e-4 `
    --weight_decay 1e-4 `
    --warmup_epochs 3 `
    --early_stop_patience 8 `
    --patch_topk 3 `
    --model_name global_local_high_perf `
    --seed 42

# 预期性能:
# - 准确率: ~93-94%
# - 显存: ~14GB
# - 速度: ~2.5s/batch
# - 参数量: 50.4M

Write-Host "High Performance configuration training started..." -ForegroundColor Green
Write-Host "Expected accuracy: ~93-94% (best)" -ForegroundColor Cyan
Write-Host "VRAM usage: ~14GB (requires high-end GPU)" -ForegroundColor Yellow
