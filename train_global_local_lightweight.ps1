# Global-Local Dual Stream - Lightweight 配置
# 适用于资源受限环境

python src/train.py `
    --use_global_local `
    --use_learnable_srm `
    --srm_out_channels 8 `
    --srm_kernel_size 3 `
    --srm_lr_scale 0.1 `
    --fusion_mode replace `
    --global_size 320 `
    --share_backbone `
    --feature_fusion_type add `
    --num_epochs 50 `
    --batch_size 64 `
    --learning_rate 1e-4 `
    --weight_decay 1e-4 `
    --warmup_epochs 3 `
    --early_stop_patience 8 `
    --patch_topk 1 `
    --model_name global_local_lightweight `
    --seed 42

# 预期性能:
# - 准确率: ~91-92%
# - 显存: ~5GB
# - 速度: ~0.9s/batch
# - 参数量: 23.5M

Write-Host "Lightweight configuration training started..." -ForegroundColor Green
Write-Host "Expected accuracy: ~91-92%" -ForegroundColor Cyan
Write-Host "VRAM usage: ~5GB (memory efficient)" -ForegroundColor Cyan
