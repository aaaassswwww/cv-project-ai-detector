import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import logging

from torchvision import transforms, models
import matplotlib.pyplot as plt

from dataset import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.patch import patch_img, patch_img_deterministic
from networks.ssp import ssp
from utils.util import set_seed, init_logger, make_worker_init_fn
from utils.loss import build_loss
from utils.transform import build_train_transform, build_val_transform


def collate_fn_dual_stream(batch):
    """
    自定义 collate 函数，用于处理 DualStreamTransform 返回的字典数据
    
    处理不同数量的 patches：由于 var_thresh 过滤，不同图片可能返回不同数量的 patches
    通过 padding 统一到最大的 K
    
    Args:
        batch: List[Tuple[Dict, label]]
            每个元素是 (data_dict, label)
            data_dict = {'local': (K_i, 3, 256, 256), 'global': (3, H, W)}
            注意：K_i 可能因图片而异
    
    Returns:
        data_dict, labels
            data_dict = {'local': (B, K_max, 3, 256, 256), 'global': (B, 3, H, W)}
            labels: (B,)
    """
    data_dicts, labels = zip(*batch)
    
    # 找出最大的 patch 数量
    max_k = max(d['local'].shape[0] for d in data_dicts)
    
    # Pad local patches 到相同的 K
    # List[(K_i, 3, 256, 256)] -> (B, K_max, 3, 256, 256)
    padded_locals = []
    for d in data_dicts:
        local = d['local']  # (K_i, 3, 256, 256)
        k_i = local.shape[0]
        
        if k_i < max_k:
            # 需要 padding：复制最后一个 patch 来填充
            # 这样比用零填充更合理，避免引入全黑的 patch
            padding_needed = max_k - k_i
            last_patch = local[-1:].repeat(padding_needed, 1, 1, 1)  # (padding_needed, 3, 256, 256)
            local = torch.cat([local, last_patch], dim=0)  # (K_max, 3, 256, 256)
        
        padded_locals.append(local)
    
    local_patches = torch.stack(padded_locals, dim=0)  # (B, K_max, 3, 256, 256)
    
    # Stack global images: List[(3, H, W)] -> (B, 3, H, W)
    global_images = torch.stack([d['global'] for d in data_dicts], dim=0)
    
    # Stack labels: List[int] -> (B,)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {'local': local_patches, 'global': global_images}, labels


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--patch_size', default=32, type=int)
parser.add_argument('--split', default='train', type=str)
parser.add_argument('--label_smoothing', default=0.1, type=float, help='label smoothing factor in [0,1]')
parser.add_argument('--jpeg_p_global', default=0.2, type=float, help='probability of JPEG compression before patching')
parser.add_argument('--jpeg_p_patch', default=0.05, type=float, help='probability of JPEG compression after patching')
parser.add_argument('--jpeg_quality_min', default=30, type=int, help='min JPEG quality for compression augment')
parser.add_argument('--jpeg_quality_max', default=95, type=int, help='max JPEG quality for compression augment')
parser.add_argument('--blur_p', default=0.15, type=float, help='probability of gaussian blur')
parser.add_argument('--blur_kernel_size', default=3, type=int, help='gaussian blur kernel size (odd integer)')
parser.add_argument('--blur_sigma_min', default=0.1, type=float, help='min gaussian sigma')
parser.add_argument('--blur_sigma_max', default=2.0, type=float, help='max gaussian sigma')
parser.add_argument('--resample_p', default=0.15, type=float, help='probability of downsample-upsample resample')
parser.add_argument('--resample_scale_min', default=0.5, type=float, help='min scale factor for downsample (0,1]')
parser.add_argument('--resample_scale_max', default=0.9, type=float, help='max scale factor for downsample (0,1]')
parser.add_argument('--noise_p', default=0.1, type=float, help='probability of gaussian noise')
parser.add_argument('--noise_sigma_min', default=0.005, type=float, help='min gaussian noise sigma in [0,1] space')
parser.add_argument('--noise_sigma_max', default=0.02, type=float, help='max gaussian noise sigma in [0,1] space')
parser.add_argument('--freq_p', default=0.1, type=float, help='probability of frequency perturbation')
parser.add_argument('--freq_scale_min', default=0.7, type=float, help='min scale for high-frequency components')
parser.add_argument('--freq_scale_max', default=1.3, type=float, help='max scale for high-frequency components')
parser.add_argument('--freq_radius', default=0.25, type=float, help='radius threshold for high-frequency scaling (0-0.5]')
parser.add_argument('--patch_var_thresh', default=5.0, type=float, help='variance threshold to drop flat patches (0 to disable)')
parser.add_argument('--patch_topk', default=3, type=int, help='top-K simplest patches to use (>=1)')

# Learnable SRM 相关参数
parser.add_argument('--use_learnable_srm', action='store_true', help='use learnable SRM instead of classic SRMConv2d')
parser.add_argument('--srm_out_channels', default=12, type=int, help='output channels for learnable SRM (default 12)')
parser.add_argument('--srm_kernel_size', default=5, type=int, help='kernel size for learnable SRM (3 or 5)')
parser.add_argument('--srm_block_type', default='srm_only', type=str, help='SRM block type: srm_only, srm_bn, srm_bn_relu, srm_bn_leaky')
parser.add_argument('--srm_freeze_epochs', default=0, type=int, help='freeze learnable SRM for N epochs (0 to not freeze)')
parser.add_argument('--srm_lr_scale', default=1.0, type=float, help='learning rate scale for SRM (e.g., 0.1 for 10%% of base lr)')
parser.add_argument('--srm_use_norm', action='store_true', help='use GroupNorm after SRM for output stabilization (recommended)')
parser.add_argument('--srm_use_mixing', action='store_true', help='use 1x1 conv for channel mixing (learnable kernel fusion)')
parser.add_argument('--srm_seed', default=42, type=int, help='random seed for SRM kernel perturbation (use different values for ensemble)')
parser.add_argument('--fusion_mode', default='replace', type=str, choices=['replace', 'concat', 'dual_stream'], help='fusion mode: replace (SRM only), concat (SRM+RGB), dual_stream (SRM stream + RGB stream)')

# Global-Local Dual Stream 相关参数
parser.add_argument('--use_global_local', action='store_true', help='use Global-Local Dual Stream architecture')
parser.add_argument('--global_size', default=384, type=int, help='global stream input size (320, 384, or 512)')
parser.add_argument('--share_backbone', action='store_true', help='share ResNet backbone between local and global streams (lightweight mode)')
parser.add_argument('--feature_fusion_type', default='concat', type=str, choices=['concat', 'add', 'attention'], help='feature fusion type: concat, add (weighted), attention')

parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='/shared-nvme/outputs', type=str, help='output directory')
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup epochs')
parser.add_argument('--early_stop_patience', default=8, type=int, help='early stopping patience (epochs)')
parser.add_argument('--min_delta', default=1e-4, type=float, help='minimum val loss improvement to reset patience')
parser.add_argument('--eta_min', default=1e-6, type=float, help='minimum lr for cosine annealing')

args = parser.parse_args()

# 定义 patch 提取函数
train_patch_fn = lambda img: patch_img(
    img,
    args.patch_size,
    256,
    deterministic=False,
    var_thresh=args.patch_var_thresh,
    topk=args.patch_topk,
)

val_patch_fn = lambda img: patch_img_deterministic(
    img,
    args.patch_size,
    256,
    var_thresh=args.patch_var_thresh,
    topk=args.patch_topk,
)

# 使用自定义 Transform 类构建清晰的 pipeline
# 如果使用 global_local，需要特殊的 transform 同时返回全局图和 patch
if hasattr(args, 'use_global_local') and args.use_global_local:
    from utils.transform import build_dual_stream_train_transform, build_dual_stream_val_transform
    train_transform = build_dual_stream_train_transform(args, train_patch_fn)
    val_transform = build_dual_stream_val_transform(args, val_patch_fn)
else:
    train_transform = build_train_transform(args, train_patch_fn)
    val_transform = build_val_transform(args, val_patch_fn)

def build_resnet50_model(num_classes=2):
    """构建不使用预训练权重的ResNet50模型"""
    
    # 创建ResNet50模型，pretrained=False表示不使用预训练权重
    model = models.resnet50(pretrained=False)
    
    # 修改最后的全连接层以适应二分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # 打印模型结构信息
    # print("模型结构:")
    # print(f"  - 输入尺寸: 224x224 RGB图像")
    # print(f"  - 输出类别: {num_classes} (0: 真实, 1: AI生成)")
    # print(f"  - 总参数数量: {sum(p.numel() for p in model.parameters())}")
    # print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, topk, use_global_local=False):
    """训练一个epoch - 带tqdm进度条
    
    注意：训练时使用 Patch-level loss 和准确率
    - 每个 patch 独立计算 loss，增加有效 batch size
    - 训练指标是 patch-level，仅供参考
    - 真实性能以验证集的 Image-level 准确率为准
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [训练]', 
                unit='batch', leave=True)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        labels = labels.to(device)
        
        # 处理 global_local 模型的字典输入
        if use_global_local:
            # inputs 是字典 {'local': (B, K, 3, 256, 256), 'global': (B, 3, H, W)}
            inputs_local = inputs['local']  # (B, K, 3, 256, 256)
            inputs_global = inputs['global'].to(device)  # (B, 3, global_size, global_size)
            
            # 展平 local patches
            b, k, c, h, w = inputs_local.shape
            inputs_local = inputs_local.view(b * k, c, h, w).to(device)  # (B*K, 3, 256, 256)
            
            # 扩展标签
            labels_expanded = labels.view(-1, 1).repeat(1, k).view(-1)  # (B*K,)
            
            # 前向传播（传入两个参数）
            optimizer.zero_grad()
            outputs = model(inputs_local, inputs_global).view(-1)  # (B*K,)
            loss = criterion(outputs, labels_expanded.float())
        else:
            # 原有的单流模型逻辑
            if inputs.dim() == 5:
                b, k, c, h, w = inputs.shape
                inputs = inputs.view(b * k, c, h, w)
                labels_expanded = labels.view(-1, 1).repeat(1, k).view(-1)
            else:
                inputs = inputs
                labels_expanded = labels
            inputs = inputs.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)  # 确保输出为 (N,) 形状
            loss = criterion(outputs, labels_expanded.float())
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        probs = torch.sigmoid(outputs)  # outputs 已经是 (N,)
        predicted = (probs > 0.5).long()
        total += labels_expanded.size(0)
        correct += predicted.eq(labels_expanded).sum().item()
        
        # 更新进度条显示
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'AvgLoss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs, topk, use_global_local=False):
    """验证一个epoch - 带tqdm进度条
    
    重要：计算 Image-level 准确率（非 Patch-level）
    - 对每张图的 K 个 patch 预测结果进行软投票（平均概率）
    - 与原始图像标签对比，计算真实的准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于存储所有预测和标签以计算详细指标（Image-level）
    all_preds = []
    all_labels = []
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [验证]', 
                unit='batch', leave=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device)  # (B,)
            batch_size = labels.size(0)
            
            # 处理 global_local 模型的字典输入
            if use_global_local:
                # inputs 是字典
                inputs_local = inputs['local']  # (B, K, 3, 256, 256)
                inputs_global = inputs['global'].to(device)  # (B, 3, H, W)
                
                # 展平 local patches
                b, k, c, h, w = inputs_local.shape
                inputs_local = inputs_local.view(b * k, c, h, w).to(device)  # (B*K, 3, 256, 256)
                
                # 扩展标签用于 loss 计算（patch-level loss）
                labels_expanded = labels.view(-1, 1).repeat(1, k).view(-1)  # (B*K,)
                
                # 前向传播
                outputs = model(inputs_local, inputs_global)  # (B*K, 1)
                loss = criterion(outputs.view(-1), labels_expanded.float())
                
                # ===== Image-level 准确率计算 =====
                # 重塑为 (B, K, 1)
                outputs_per_image = outputs.view(b, k, 1)  # (B, K, 1)
                probs_per_image = torch.sigmoid(outputs_per_image)  # (B, K, 1)
                
                # 软投票：对每张图的 K 个 patch 概率求平均
                avg_probs = probs_per_image.mean(dim=1).squeeze()  # (B,)
                predicted = (avg_probs > 0.5).long()  # (B,)
                
            else:
                # 原有逻辑
                if inputs.dim() == 5:  # (B, K, C, H, W)
                    b, k, c, h, w = inputs.shape
                    inputs = inputs.view(b * k, c, h, w).to(device)  # (B*K, C, H, W)
                    labels_expanded = labels.view(-1, 1).repeat(1, k).view(-1)  # (B*K,)
                    
                    # 前向传播
                    outputs = model(inputs)  # (B*K, 1)
                    loss = criterion(outputs.view(-1), labels_expanded.float())
                    
                    # ===== Image-level 准确率计算 =====
                    outputs_per_image = outputs.view(b, k, 1)  # (B, K, 1)
                    probs_per_image = torch.sigmoid(outputs_per_image)  # (B, K, 1)
                    avg_probs = probs_per_image.mean(dim=1).squeeze()  # (B,)
                    predicted = (avg_probs > 0.5).long()  # (B,)
                    
                else:  # (B, C, H, W) - topk=1
                    inputs = inputs.to(device)
                    outputs = model(inputs).view(-1)  # (B,)
                    loss = criterion(outputs, labels.float())
                    
                    # Image-level（单 patch 情况）
                    probs = torch.sigmoid(outputs)  # (B,)
                    predicted = (probs > 0.5).long()  # (B,)
            
            # 统计信息（Image-level）
            running_loss += loss.item()
            total += batch_size  # 注意：这里是图像数量，不是 patch 数量
            correct += predicted.eq(labels).sum().item()
            
            # 保存预测结果（Image-level）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条（修复：使用 batch_idx + 1）
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'AvgLoss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # 计算详细指标
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                   target_names=['真实', 'AI生成'], 
                                   output_dict=True)
    
    return epoch_loss, epoch_acc, cm, report

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # ===== 创建保存目录（带详细验证） =====
    print(f"\n{'='*70}")
    print("初始化保存目录...")
    print(f"{'='*70}")
    
    # 创建输出目录
    output_dir_abs = os.path.abspath(args.output_dir)
    print(f"Output directory: {output_dir_abs}")
    os.makedirs(output_dir_abs, exist_ok=True)
    if os.path.exists(output_dir_abs):
        print(f"✓ Output directory exists")
    else:
        print(f"❌ Failed to create output directory!")
    
    # 创建模型目录
    model_dir = os.path.join(output_dir_abs, args.model_name)
    model_dir_abs = os.path.abspath(model_dir)
    print(f"Model directory: {model_dir_abs}")
    os.makedirs(model_dir_abs, exist_ok=True)
    if os.path.exists(model_dir_abs):
        print(f"✓ Model directory exists")
        # 测试写权限
        test_file = os.path.join(model_dir_abs, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✓ Write permission OK")
        except Exception as e:
            print(f"❌ Write permission ERROR: {e}")
    else:
        print(f"❌ Failed to create model directory!")
    
    # 初始化日志
    log_path = os.path.join(model_dir_abs, 'train.log')
    print(f"Log file: {log_path}")
    logger = init_logger(log_path)
    
    # 验证日志文件是否创建
    if os.path.exists(log_path):
        print(f"✓ Log file created successfully")
        print(f"  Size: {os.path.getsize(log_path)} bytes")
    else:
        print(f"❌ Log file not created!")
    
    print(f"{'='*70}\n")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    
    # 创建数据集和数据加载器
    logger.info("="*50)
    logger.info("Loading datasets...")
    train_dataset = Dataset(args=args, split='train',transforms=train_transform)
    
    val_dataset = Dataset(args=args,split='val',transforms=val_transform)
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    worker_init_fn = make_worker_init_fn(args.seed)

    # 根据是否使用 global_local 选择 collate_fn
    use_custom_collate = hasattr(args, 'use_global_local') and args.use_global_local
    custom_collate_fn = collate_fn_dual_stream if use_custom_collate else None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=custom_collate_fn,
    )
    
    logger.info(f"Train set: {len(train_dataset)} images | Val set: {len(val_dataset)} images")
    
    # 检查数据均衡性
    real_count = train_dataset.labels.count(0)
    fake_count = train_dataset.labels.count(1)
    logger.info(f"Train class balance - Real: {real_count}, AI-generated: {fake_count}")
    
    # 构建模型
    logger.info("="*50)
    logger.info("Building model...")
    
    # 构建 learnable SRM 配置
    learnable_srm_config = None
    if args.use_learnable_srm:
        learnable_srm_config = {
            'out_channels': args.srm_out_channels,
            'kernel_size': args.srm_kernel_size,
            'block_type': args.srm_block_type,
            'freeze_init_epochs': args.srm_freeze_epochs,
            'use_norm': args.srm_use_norm,
            'use_mixing': args.srm_use_mixing,
            'seed': args.srm_seed,
        }
        logger.info(f"  - Using Learnable SRM with config: {learnable_srm_config}")
        logger.info(f"  - Fusion mode: {args.fusion_mode}")
    else:
        logger.info("  - Using classic SRMConv2d")
        logger.info(f"  - Fusion mode: {args.fusion_mode}")
    
    # 选择模型架构
    if args.use_global_local:
        from networks.global_local import GlobalLocalDualStream
        logger.info("  - Using Global-Local Dual Stream architecture")
        logger.info(f"    * Global size: {args.global_size}x{args.global_size}")
        logger.info(f"    * Share backbone: {args.share_backbone}")
        logger.info(f"    * Feature fusion: {args.feature_fusion_type}")
        
        model = GlobalLocalDualStream(
            pretrain=False,  # 不使用预训练权重
            topk=args.patch_topk,
            use_learnable_srm=args.use_learnable_srm,
            learnable_srm_config=learnable_srm_config,
            fusion_mode=args.fusion_mode,
            global_size=args.global_size,
            share_backbone=args.share_backbone,
            fusion_type=args.feature_fusion_type,
        )
    else:
        logger.info("  - Using Single-Stream SSP architecture")
        model = ssp(
            pretrain=False,  # 不使用预训练权重
            topk=args.patch_topk,
            use_learnable_srm=args.use_learnable_srm,
            learnable_srm_config=learnable_srm_config,
            fusion_mode=args.fusion_mode,
        )
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    bce = build_loss(smoothing=args.label_smoothing)
    
    # 使用分离的参数组创建 optimizer（更稳健的方式）
    param_groups = model.get_param_groups(
        base_lr=args.learning_rate,
        srm_lr_scale=args.srm_lr_scale,
        weight_decay=args.weight_decay
    )
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    if args.use_learnable_srm and args.srm_lr_scale != 1.0:
        logger.info(f"  - SRM learning rate scale: {args.srm_lr_scale}x (lr={args.learning_rate * args.srm_lr_scale:.6e})")
    
    # 学习率调度器: warmup + 余弦退火
    cosine_t_max = max(1, args.num_epochs - args.warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.eta_min
    )
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    # 输出所有保存路径
    logger.info("="*50)
    logger.info("Output paths:")
    logger.info(f"  - Model directory: {os.path.abspath(model_dir)}")
    logger.info(f"  - Training log: {os.path.abspath(os.path.join(model_dir, 'train.log'))}")
    logger.info(f"  - Best model: {os.path.abspath(os.path.join(model_dir, 'ai-detector_best.pth'))}")
    logger.info(f"  - Checkpoints: {os.path.abspath(model_dir)}/checkpoint_epoch_*.pth")
    logger.info(f"  - Training curve: {os.path.abspath('training_history.png')}")
    logger.info("="*50)
    logger.info("Start training...")
    best_val_acc = 0.0
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # 创建epoch级别的进度条
    epoch_pbar = tqdm(range(args.num_epochs), desc="总进度", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # SRM freeze/unfreeze 管理
        if args.use_learnable_srm and args.srm_freeze_epochs > 0:
            if epoch < args.srm_freeze_epochs:
                model.freeze_srm()
                srm_status = "FROZEN"
            else:
                if epoch == args.srm_freeze_epochs:  # 解冻的第一个 epoch
                    model.unfreeze_srm()
                    logger.info(f"⚡ Unfroze SRM at epoch {epoch+1}")
                srm_status = "TRAINABLE"
        else:
            srm_status = "N/A" if not args.use_learnable_srm else "TRAINABLE"
        
        # Warmup: 线性提升到基础学习率
        if epoch < args.warmup_epochs:
            warmup_lr = args.learning_rate * (epoch + 1) / max(1, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            logger.info(f"Warmup epoch {epoch+1}/{args.warmup_epochs} | LR {warmup_lr:.6f} | SRM: {srm_status}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, bce, optimizer, device, epoch, args.num_epochs, args.patch_topk,
            use_global_local=args.use_global_local if hasattr(args, 'use_global_local') else False
        )
        
        # 验证
        val_loss, val_acc, cm, report = validate_epoch(
            model, val_loader, bce, device, epoch, args.num_epochs, args.patch_topk,
            use_global_local=args.use_global_local if hasattr(args, 'use_global_local') else False
        )
        
        # 更新学习率（warmup 之后再使用余弦退火）
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        # 更新总进度条
        epoch_pbar.set_postfix({
            'TrainAcc': f'{train_acc:.2f}%',
            'ValAcc': f'{val_acc:.2f}%',
            'ValLoss': f'{val_loss:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # 打印详细结果
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
            f"LR {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time {epoch_time:.1f}s"
        )
        logger.info(f"Confusion matrix:\n{cm}")
        
        # 保存最佳模型（以验证损失为准，支持早停）
        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve_epochs = 0
            best_path = os.path.join(model_dir_abs, 'ai-detector_best.pth')
            
            # 保存前验证目录
            if not os.path.exists(os.path.dirname(best_path)):
                logger.warning(f"⚠ Directory doesn't exist, creating: {os.path.dirname(best_path)}")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
            
            # 保存模型
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'args': vars(args),
                }, best_path)
                
                # 验证文件确实被创建
                if os.path.exists(best_path):
                    file_size = os.path.getsize(best_path) / (1024 * 1024)  # MB
                    logger.info(f"✓ Saved best model (ValLoss {val_loss:.4f}, ValAcc {val_acc:.2f}%)")
                    logger.info(f"  → {os.path.abspath(best_path)}")
                    logger.info(f"  → Size: {file_size:.2f} MB")
                else:
                    logger.error(f"❌ File save claimed success but file doesn't exist!")
                    logger.error(f"  → Attempted path: {os.path.abspath(best_path)}")
            except Exception as e:
                logger.error(f"❌ Failed to save model: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            no_improve_epochs += 1
            logger.info(f"↻ No significant val improvement ({no_improve_epochs}/{args.early_stop_patience})")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir_abs, f'checkpoint_epoch_{epoch+1}.pth')
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'args': vars(args),
                }, checkpoint_path)
                
                # 验证文件
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                    logger.info(f"✓ Saved checkpoint (epoch {epoch+1})")
                    logger.info(f"  → {os.path.abspath(checkpoint_path)}")
                    logger.info(f"  → Size: {file_size:.2f} MB")
                else:
                    logger.error(f"❌ Checkpoint save failed - file doesn't exist!")
            except Exception as e:
                logger.error(f"❌ Failed to save checkpoint: {e}")

        # 早停
        if no_improve_epochs >= args.early_stop_patience:
            logger.info(f"⚑ Early stopping: {no_improve_epochs} epochs without min_delta {args.min_delta}")
            break
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 最终报告
    logger.info("="*50)
    logger.info("Training finished!")
    logger.info(f"Best val accuracy: {best_val_acc:.2f}%")
    logger.info("="*50)
    logger.info("Saved files:")
    
    # 检查并列出所有文件
    best_model_path = os.path.join(model_dir_abs, 'ai-detector_best.pth')
    log_path = os.path.join(model_dir_abs, 'train.log')
    curve_path = os.path.abspath('training_history.png')
    
    if os.path.exists(best_model_path):
        size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
        logger.info(f"  ✓ Best model: {best_model_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"  ❌ Best model NOT FOUND: {best_model_path}")
    
    if os.path.exists(log_path):
        size_kb = os.path.getsize(log_path) / 1024
        logger.info(f"  ✓ Training log: {log_path} ({size_kb:.2f} KB)")
    else:
        logger.warning(f"  ❌ Training log NOT FOUND: {log_path}")
    
    if os.path.exists(curve_path):
        size_kb = os.path.getsize(curve_path) / 1024
        logger.info(f"  ✓ Training curve: {curve_path} ({size_kb:.2f} KB)")
    else:
        logger.warning(f"  ❌ Training curve NOT FOUND: {curve_path}")
    
    # 列出所有保存的checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(model_dir_abs, 'checkpoint_epoch_*.pth'))
    if checkpoints:
        logger.info(f"  ✓ Checkpoints ({len(checkpoints)}):")
        for ckpt in sorted(checkpoints):
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            logger.info(f"    * {ckpt} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"  ❌ No checkpoints found in {model_dir_abs}")
    
    # 列出目录中的所有文件
    logger.info(f"\n  Directory contents: {model_dir_abs}")
    try:
        all_files = os.listdir(model_dir_abs)
        if all_files:
            for f in sorted(all_files):
                f_path = os.path.join(model_dir_abs, f)
                if os.path.isfile(f_path):
                    size = os.path.getsize(f_path)
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.2f} MB"
                    else:
                        size_str = f"{size / 1024:.2f} KB"
                    logger.info(f"    - {f} ({size_str})")
        else:
            logger.warning(f"    Directory is empty!")
    except Exception as e:
        logger.error(f"    Failed to list directory: {e}")
    
    logger.info("="*50)
    
    return model, history

def plot_training_history(history):
    """绘制训练过程中的损失和准确率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    # plt.show()

if __name__ == "__main__":
    print("="*60)
    print("AI生成图像检测器 - ResNet50训练框架")
    print("="*60)
    
    start_time = time.time()
    trained_model, training_history = main()
    total_time = time.time() - start_time

    logger = logging.getLogger("train")
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    logger.info("Training history saved to training_history.png")
    
