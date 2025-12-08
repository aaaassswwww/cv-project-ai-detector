import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import argparse
import time

from torchvision import transforms, models
import matplotlib.pyplot as plt

from dataset import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from patch import patch_img
from networks.ssp import ssp


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--patch_size', default=32, type=int)
parser.add_argument('--split', default='train', type=str)

parser.add_argument('--seed', default=666, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='checkpoints', type=str, help='output directory')
parser.add_argument('--model_name', default='ssp', type=str)

args = parser.parse_args()

patch_fun = transforms.Lambda(
        lambda img: patch_img(img, args.patch_size, 256)
        )

train_transform = transforms.Compose([
    patch_fun,
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet统计量
])

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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch - 带tqdm进度条"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [训练]', 
                unit='batch', leave=True)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.ravel(), labels.float())
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        probs = torch.sigmoid(outputs).squeeze()
        predicted = (probs > 0.5).long()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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


def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
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
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.ravel(), labels.float())
            
            # 统计信息
            running_loss += loss.item()

            probs = torch.sigmoid(outputs).squeeze()
            predicted = (probs > 0.5).long()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 保存预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            avg_loss = running_loss / len(pbar)
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
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # 创建数据集和数据加载器
    print("\n" + "="*50)
    print("加载数据集...")
    train_dataset = Dataset(args=args, split='train',transforms=train_transform)
    
    val_dataset = Dataset(args=args,split='val',transforms=train_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
    )
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    # 检查数据均衡性
    real_count = train_dataset.labels.count(0)
    fake_count = train_dataset.labels.count(1)
    print(f"训练集类别分布 - 真实: {real_count}, AI生成: {fake_count}")
    
    # 构建模型
    print("\n" + "="*50)
    print("构建模型...")
    # model = build_resnet50_model(num_classes=2)
    model = ssp()
    model = model.to(device)
    
    # 定义损失函数和优化器
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    # 训练循环
    print("\n" + "="*50)
    print("开始训练...")
    best_val_acc = 0.0
    
    # 创建epoch级别的进度条
    epoch_pbar = tqdm(range(args.num_epochs), desc="总进度", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, bce, optimizer, device, epoch, args.num_epochs
        )
        
        # 验证
        val_loss, val_acc, cm, report = validate_epoch(
            model, val_loader, bce, device, epoch, args.num_epochs
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
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
        print(f"\nEpoch {epoch+1}/{args.num_epochs} 结果:")
        print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        print(f"  混淆矩阵:\n{cm}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  耗时: {epoch_time:.1f}秒")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, args.model_name)
            model_path = os.path.join(model_path, 'ai-detector_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, model_path)
            print(f"  ✓ 保存最佳模型到: {model_path} (准确率: {val_acc:.2f}%)")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, args.model_name)
            checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, checkpoint_path)
            print(f"  ✓ 保存检查点到: {checkpoint_path}")
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 最终报告
    print("\n" + "="*50)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, history

def plot_training_history(history):
    """绘制训练过程中的损失和准确率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练和验证准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("="*60)
    print("AI生成图像检测器 - ResNet50训练框架")
    print("="*60)
    
    # 运行主训练流程
    start_time = time.time()
    trained_model, training_history = main()
    total_time = time.time() - start_time
    
    print(f"\n总训练时间: {total_time/60:.1f} 分钟")
    print("训练历史已保存到 training_history_detailed.png")
    
