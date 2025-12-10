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


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--patch_size', default=32, type=int)
parser.add_argument('--split', default='train', type=str)
parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing factor in [0,1]')
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

parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='checkpoints', type=str, help='output directory')
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, topk):
    """训练一个epoch - 带tqdm进度条"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [训练]', 
                unit='batch', leave=True)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # inputs shape: (B, K, C, H, W) if topk>1
        labels = labels.to(device)
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


def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs, topk):
    """验证一个epoch - 带tqdm进度条"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于存储所有预测和标签以计算详细指标
    all_preds = []
    all_labels = []
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [验证]', 
                unit='batch', leave=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device)
            if inputs.dim() == 5:
                b, k, c, h, w = inputs.shape
                inputs = inputs.view(b * k, c, h, w)
                labels_expanded = labels.view(-1, 1).repeat(1, k).view(-1)
            else:
                labels_expanded = labels
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs).view(-1)  # 确保输出为 (N,) 形状
            loss = criterion(outputs, labels_expanded.float())
            
            # 统计信息
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)  # outputs 已经是 (N,)
            predicted = (probs > 0.5).long()
            total += labels_expanded.size(0)
            correct += predicted.eq(labels_expanded).sum().item()
            
            # 保存预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_expanded.cpu().numpy())
            
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
    
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    logger = init_logger(os.path.join(model_dir, 'train.log'))
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

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    
    logger.info(f"Train set: {len(train_dataset)} images | Val set: {len(val_dataset)} images")
    
    # 检查数据均衡性
    real_count = train_dataset.labels.count(0)
    fake_count = train_dataset.labels.count(1)
    logger.info(f"Train class balance - Real: {real_count}, AI-generated: {fake_count}")
    
    # 构建模型
    logger.info("="*50)
    logger.info("Building model...")
    # model = build_resnet50_model(num_classes=2)
    model = ssp()
    model = model.to(device)
    
    # 定义损失函数和优化器
    bce = build_loss(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
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
    
    # 训练循环
    logger.info("="*50)
    logger.info("Start training...")
    best_val_acc = 0.0
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # 创建epoch级别的进度条
    epoch_pbar = tqdm(range(args.num_epochs), desc="总进度", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Warmup: 线性提升到基础学习率
        if epoch < args.warmup_epochs:
            warmup_lr = args.learning_rate * (epoch + 1) / max(1, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            logger.info(f"Warmup epoch {epoch+1}/{args.warmup_epochs} | LR {warmup_lr:.6f}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, bce, optimizer, device, epoch, args.num_epochs, args.patch_topk
        )
        
        # 验证
        val_loss, val_acc, cm, report = validate_epoch(
            model, val_loader, bce, device, epoch, args.num_epochs, args.patch_topk
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
            best_path = os.path.join(model_dir, 'ai-detector_best.pth')
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
            logger.info(f"✓ Saved best model to: {best_path} (ValLoss {val_loss:.4f}, ValAcc {val_acc:.2f}%)")
        else:
            no_improve_epochs += 1
            logger.info(f"↻ No significant val improvement ({no_improve_epochs}/{args.early_stop_patience})")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
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
            logger.info(f"✓ Saved checkpoint to: {checkpoint_path}")

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
    
