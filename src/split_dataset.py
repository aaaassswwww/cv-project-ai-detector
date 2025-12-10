"""
Train/Val 数据集划分脚本

从 dataset/train 中随机抽取固定数量的图片移动到 dataset/val
使用固定的随机种子确保结果可复现

使用方法:
    python split_dataset.py --seed 42 --num_per_class 800
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(
    dataset_root: str = "./dataset",
    num_per_class: int = 400,
    seed: int = 42,
    dry_run: bool = False
):
    """
    从 train 集中抽取图片到 val 集
    
    Args:
        dataset_root: 数据集根目录
        num_per_class: 每个类别抽取的图片数量
        seed: 随机种子（确保可复现）
        dry_run: 如果为 True，只打印不实际移动
    """
    random.seed(seed)
    
    train_dir = Path(dataset_root) / "train"
    val_dir = Path(dataset_root) / "val"
    
    classes = ["0_real", "1_fake"]
    
    print(f"Dataset root: {dataset_root}")
    print(f"Random seed: {seed}")
    print(f"Images per class: {num_per_class}")
    print(f"Dry run: {dry_run}")
    print("=" * 50)
    
    for cls in classes:
        train_cls_dir = train_dir / cls
        val_cls_dir = val_dir / cls
        
        # 确保验证集目录存在
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取训练集中该类别的所有图片
        if not train_cls_dir.exists():
            print(f"Warning: {train_cls_dir} does not exist, skipping...")
            continue
        
        all_images = sorted([f for f in os.listdir(train_cls_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])
        
        print(f"\nClass: {cls}")
        print(f"  Total images in train: {len(all_images)}")
        
        if len(all_images) < num_per_class:
            print(f"  Warning: Only {len(all_images)} images available, "
                  f"cannot sample {num_per_class}!")
            selected = all_images
        else:
            # 固定种子随机抽样
            selected = random.sample(all_images, num_per_class)
        
        print(f"  Selected for val: {len(selected)}")
        
        # 移动文件
        moved_count = 0
        for img_name in selected:
            src = train_cls_dir / img_name
            dst = val_cls_dir / img_name
            
            if dry_run:
                print(f"    [DRY RUN] Would move: {src} -> {dst}")
            else:
                if src.exists():
                    shutil.move(str(src), str(dst))
                    moved_count += 1
                else:
                    print(f"    Warning: {src} not found, skipping...")
        
        if not dry_run:
            print(f"  Moved: {moved_count} images")
    
    print("\n" + "=" * 50)
    print("Dataset split completed!")
    
    # 打印最终统计
    print("\nFinal statistics:")
    for split in ["train", "val"]:
        split_dir = Path(dataset_root) / split
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                count = len([f for f in os.listdir(cls_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])
                print(f"  {split}/{cls}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train dataset into train/val")
    parser.add_argument('--dataset_root', default='./dataset', type=str,
                        help='Dataset root directory')
    parser.add_argument('--num_per_class', default=800, type=int,
                        help='Number of images to move per class')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, only print what would be done without moving files')
    
    args = parser.parse_args()
    
    split_dataset(
        dataset_root=args.dataset_root,
        num_per_class=args.num_per_class,
        seed=args.seed,
        dry_run=args.dry_run
    )
