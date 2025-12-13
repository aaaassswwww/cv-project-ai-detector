"""
使用 TTA 进行模型评估

用法:
    python src/eval_with_tta.py --tta_mode standard
    python src/eval_with_tta.py --tta_mode full --split test
    python src/eval_with_tta.py --tta_mode simple --split val
"""

from PIL import Image
import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from networks.ssp import ssp
from dataset import Dataset
from torch.utils.data import DataLoader
from utils.patch import patch_img_deterministic
from utils.util import set_seed
from utils.tta import create_tta_predictor
from utils.transform import PatchTransform
from natsort import natsorted
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time

parser = argparse.ArgumentParser(description='使用 TTA 进行模型评估')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--split', default='test', type=str, help="dataset split in ['val', 'test']")
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--model_path', default="./shared-nvme/ssp-models", help='Pretrained Model Path')
parser.add_argument('--output_file', default=None, help='CSV output file (default: result_tta_{mode}.csv)')
parser.add_argument('--batch_size', default=1, type=int, help='batch size (TTA uses single image)')
parser.add_argument('--seed', default=42, type=int, help='random seed')

# TTA 参数
parser.add_argument('--tta_mode', default='full', type=str, 
                    choices=['full', 'standard', 'simple', 'minimal'],
                    help='TTA mode: full (all), standard (recommended), simple (fast), minimal (hflip only)')
parser.add_argument('--tta_hflip', action='store_true', help='enable horizontal flip')
parser.add_argument('--tta_vflip', action='store_true', help='enable vertical flip')
parser.add_argument('--tta_rotation', action='store_true', help='enable 90° rotation')
parser.add_argument('--tta_jpeg', action='store_true', help='enable JPEG compression')
parser.add_argument('--tta_multiscale', action='store_true', help='enable multi-scale')
parser.add_argument('--tta_aggregation', default='mean', type=str, 
                    choices=['mean', 'median', 'vote'],
                    help='aggregation method for TTA predictions')

# Learnable SRM 参数（如果模型使用了）
parser.add_argument('--use_learnable_srm', action='store_true', help='model uses learnable SRM')
parser.add_argument('--fusion_mode', default='concat', type=str, 
                    choices=['replace', 'concat', 'dual_stream'],
                    help='fusion mode of the model')

# Global-Local Dual Stream 参数
parser.add_argument('--use_global_local', action='store_true', help='model uses Global-Local Dual Stream architecture')
parser.add_argument('--global_size', default=384, type=int, help='global stream input size')
parser.add_argument('--share_backbone', action='store_true', help='share ResNet backbone between streams')
parser.add_argument('--feature_fusion_type', default='concat', type=str, 
                    choices=['concat', 'add', 'attention'],
                    help='feature fusion type')

parser.add_argument('--patch_topk', default=5, type=int, help='top-K patches to use')
parser.add_argument('--patch_size', default=32, type=int, help='patch size')

args = parser.parse_args()


def load_model(model_path, device):
    """加载模型"""
    # 从 checkpoint 加载模型配置
    full_model_path = os.path.join(args.model_path, args.model_name, 'ai-detector_best.pth')
    
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model not found: {full_model_path}")
    
    checkpoint = torch.load(full_model_path, map_location=device)
    
    # 从 checkpoint 获取模型参数
    saved_args = checkpoint.get('args', {})
    
    # 获取训练时使用的 topk（关键参数！）
    saved_topk = saved_args.get('patch_topk', 1)
    if saved_topk != args.patch_topk:
        print(f"⚠ Warning: Model was trained with topk={saved_topk}, but you specified topk={args.patch_topk}")
        print(f"  → Using topk={saved_topk} from checkpoint")
        args.patch_topk = saved_topk
    
    # 构建模型
    use_learnable_srm = saved_args.get('use_learnable_srm', args.use_learnable_srm)
    fusion_mode = saved_args.get('fusion_mode', args.fusion_mode)
    use_global_local = saved_args.get('use_global_local', args.use_global_local)
    
    learnable_srm_config = None
    if use_learnable_srm:
        learnable_srm_config = {
            'out_channels': saved_args.get('srm_out_channels', 12),
            'kernel_size': saved_args.get('srm_kernel_size', 5),
            'block_type': saved_args.get('srm_block_type', 'srm_only'),
            'use_norm': saved_args.get('srm_use_norm', False),
            'use_mixing': saved_args.get('srm_use_mixing', False),
            'seed': saved_args.get('srm_seed', 42),
        }
        print(f"✓ Model uses Learnable SRM: {learnable_srm_config}")
        print(f"✓ Fusion mode: {fusion_mode}")
    
    # 根据训练时的架构选择模型
    if use_global_local:
        from networks.global_local import GlobalLocalDualStream
        
        global_size = saved_args.get('global_size', args.global_size)
        share_backbone = saved_args.get('share_backbone', args.share_backbone)
        feature_fusion_type = saved_args.get('feature_fusion_type', args.feature_fusion_type)
        
        print(f"✓ Using Global-Local Dual Stream architecture")
        print(f"  - Global size: {global_size}x{global_size}")
        print(f"  - Share backbone: {share_backbone}")
        print(f"  - Feature fusion: {feature_fusion_type}")
        
        model = GlobalLocalDualStream(
            pretrain=False,
            topk=saved_topk,
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config=learnable_srm_config,
            fusion_mode=fusion_mode,
            global_size=global_size,
            share_backbone=share_backbone,
            fusion_type=feature_fusion_type,
        )
    else:
        print(f"✓ Using Single-Stream SSP architecture")
        model = ssp(
            pretrain=False,
            topk=saved_topk,  # 使用训练时的 topk
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config=learnable_srm_config,
            fusion_mode=fusion_mode,
        )
    
    # 尝试加载模型权重，并提供详细的调试信息
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"⚠ Warning: Strict loading failed, trying with strict=False")
        print(f"  Error: {str(e)[:200]}...")
        
        # 打印调试信息
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        model_keys = set(model.state_dict().keys())
        
        missing_in_checkpoint = model_keys - checkpoint_keys
        unexpected_in_checkpoint = checkpoint_keys - model_keys
        
        if missing_in_checkpoint:
            print(f"\n  Missing in checkpoint ({len(missing_in_checkpoint)} keys):")
            for key in list(missing_in_checkpoint)[:5]:
                print(f"    - {key}")
            if len(missing_in_checkpoint) > 5:
                print(f"    ... and {len(missing_in_checkpoint) - 5} more")
        
        if unexpected_in_checkpoint:
            print(f"\n  Unexpected in checkpoint ({len(unexpected_in_checkpoint)} keys):")
            for key in list(unexpected_in_checkpoint)[:5]:
                print(f"    - {key}")
            if len(unexpected_in_checkpoint) > 5:
                print(f"    ... and {len(unexpected_in_checkpoint) - 5} more")
        
        # 使用 strict=False 加载
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"\n✓ Model loaded with strict=False")
        if result.missing_keys:
            print(f"  ⚠ {len(result.missing_keys)} missing keys (will use random init)")
        if result.unexpected_keys:
            print(f"  ⚠ {len(result.unexpected_keys)} unexpected keys (ignored)")
    
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded from: {full_model_path}")
    print(f"✓ Model epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Model val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model


def evaluate_with_tta(model, device, split):
    """使用 TTA 进行评估"""
    print("\n" + "="*60)
    print(f"开始使用 TTA 评估 (split={split}, mode={args.tta_mode})")
    print("="*60)
    
    # 准备 patch 提取函数
    def patch_fn(img):
        return patch_img_deterministic(
            img, 
            args.patch_size, 
            256, 
            var_thresh=0.0, 
            topk=args.patch_topk
        )
    
    # 准备 transform 函数
    patch_transform = PatchTransform(
        patch_size=256,
        apply_augment=False,  # 评估时不增强
        augment_config=None
    )
    
    def transform_fn(patches):
        return patch_transform(patches)
    
    # 创建 TTA 预测器
    if args.tta_mode in ['full', 'standard', 'minimal']:
        # 使用自定义参数（如果提供）
        from utils.tta import ForensicTTA
        tta_predictor = ForensicTTA(
            model=model,
            device=device,
            patch_fn=patch_fn,
            transform_fn=transform_fn,
            enable_hflip=args.tta_hflip if args.tta_hflip else (args.tta_mode != 'minimal'),
            enable_vflip=args.tta_vflip,
            enable_rotation=args.tta_rotation if args.tta_rotation else (args.tta_mode in ['full', 'standard']),
            enable_jpeg=args.tta_jpeg if args.tta_jpeg else (args.tta_mode in ['full', 'standard']),
            enable_multiscale=args.tta_multiscale if args.tta_multiscale else (args.tta_mode == 'full'),
            enable_blur=False,
            aggregation=args.tta_aggregation,
        )
    else:
        tta_predictor = create_tta_predictor(
            model=model,
            device=device,
            patch_fn=patch_fn,
            transform_fn=transform_fn,
            tta_mode=args.tta_mode,
        )
    
    # 加载数据集
    root_images = os.path.join(args.dataset_root, split)
    
    # 收集所有图像
    all_images = []
    all_labels = []
    all_image_ids = []
    
    for class_idx, class_name in enumerate(['0_real', '1_fake']):
        class_dir = os.path.join(root_images, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠ Warning: {class_dir} not found, skipping...")
            continue
        
        image_files = natsorted([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in image_files:
            all_images.append(os.path.join(class_dir, img_file))
            all_labels.append(class_idx)
            all_image_ids.append(img_file)
    
    print(f"\n✓ Found {len(all_images)} images")
    print(f"  - Real: {all_labels.count(0)}")
    print(f"  - Fake: {all_labels.count(1)}")
    
    # 预测
    predictions = []
    probabilities = []
    
    start_time = time.time()
    
    for img_path, label in tqdm(zip(all_images, all_labels), 
                                 total=len(all_images), 
                                 desc="TTA Evaluation"):
        img = Image.open(img_path).convert('RGB')
        prob = tta_predictor.predict(img)
        pred = 1 if prob > 0.5 else 0
        
        predictions.append(pred)
        probabilities.append(prob)
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary'
    )
    cm = confusion_matrix(all_labels, predictions)
    
    print("\n" + "="*60)
    print("评估结果 (with TTA)")
    print("="*60)
    print(f"Total images: {len(all_images)}")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/len(all_images):.2f}s per image)")
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # 详细分类报告
    print(f"\n详细分类报告:")
    print(classification_report(all_labels, predictions, 
                               target_names=['Real', 'Fake'], 
                               digits=4))
    
    # 保存结果
    if args.output_file is None:
        args.output_file = f"result_tta_{args.tta_mode}.csv"
    
    df = pd.DataFrame({
        'image_id': all_image_ids,
        'true_label': all_labels,
        'predicted_label': predictions,
        'probability': probabilities,
    })
    
    df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results saved to: {args.output_file}")
    
    return accuracy, precision, recall, f1, cm


def main():
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = load_model(args.model_path, device)
    
    accuracy, precision, recall, f1, cm = evaluate_with_tta(model, device, args.split)
    
    print("\n" + "="*60)
    print("✓ Evaluation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
