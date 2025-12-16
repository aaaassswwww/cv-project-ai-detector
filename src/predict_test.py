"""
对无标注测试集进行预测

用法:
    python src/predict_test.py --tta_mode none
    python src/predict_test.py --tta_mode standard --output_file my_predictions.csv
"""

from PIL import Image
import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from networks.ssp import ssp
from utils.patch import patch_img_deterministic
from utils.util import set_seed
from utils.tta import create_tta_predictor
from utils.transform import PatchTransform
from natsort import natsorted
import time

parser = argparse.ArgumentParser(description='对无标注测试集进行预测')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--test_dir', default='test', type=str, help='test directory name')
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--model_path', default="./shared-nvme/ssp-models", help='Pretrained Model Path')
parser.add_argument('--output_file', default='predictions.csv', help='CSV output file')
parser.add_argument('--seed', default=42, type=int, help='random seed')

# TTA 参数
parser.add_argument('--tta_mode', default='full', type=str, 
                    choices=['full', 'standard', 'simple', 'minimal', 'none'],
                    help='TTA mode: full (all), standard (recommended), simple (fast), minimal (hflip only), none (no TTA)')
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
    # full_model_path = os.path.join(args.model_path, args.model_name, 'checkpoint_epoch_25.pth')
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
            topk=saved_topk,
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config=learnable_srm_config,
            fusion_mode=fusion_mode,
        )
    
    # 加载模型权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"⚠ Warning: Strict loading failed, trying with strict=False")
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.missing_keys:
            print(f"  ⚠ {len(result.missing_keys)} missing keys")
        if result.unexpected_keys:
            print(f"  ⚠ {len(result.unexpected_keys)} unexpected keys")
    
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded from: {full_model_path}")
    print(f"✓ Model epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"✓ Model val_acc: {checkpoint['val_acc']:.2f}%")
    
    return model


def predict_test_set(model, device):
    """对测试集进行预测"""
    print("\n" + "="*70)
    print(f"开始预测测试集 (TTA mode={args.tta_mode})")
    print("="*70)
    
    # 检测是否是 Global-Local 模型（更可靠的检测方法）
    model_class_name = model.__class__.__name__
    is_global_local = 'GlobalLocal' in model_class_name or (hasattr(model, 'local_stream') and hasattr(model, 'global_stream'))
    
    print(f"✓ Model type: {model_class_name}")
    
    if is_global_local:
        print(f"✓ Detected Global-Local Dual Stream architecture")
        from torchvision import transforms
        # 准备全局图像的 transform
        # 注意：必须使用 ImageNet normalization（与训练时一致）
        global_transform = transforms.Compose([
            transforms.Resize((args.global_size, args.global_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        print(f"✓ Detected Single-Stream architecture")
        global_transform = None
    
    # 测试集路径
    test_path = os.path.join(args.dataset_root, args.test_dir)
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")
    
    # 收集所有图像文件
    all_images = []
    all_image_names = []
    
    # 支持直接在test目录下，或者在test的子目录下
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                all_images.append(img_path)
                # 保存相对于test_path的路径
                rel_path = os.path.relpath(img_path, test_path)
                all_image_names.append(rel_path)
    
    # 按文件名排序
    sorted_indices = np.argsort(all_image_names)
    all_images = [all_images[i] for i in sorted_indices]
    all_image_names = [all_image_names[i] for i in sorted_indices]
    
    print(f"\n✓ Found {len(all_images)} images in {test_path}")
    
    if len(all_images) == 0:
        print("❌ No images found! Please check your test directory.")
        return
    
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
        apply_augment=False,
        augment_config=None
    )
    
    def transform_fn(patches):
        return patch_transform(patches)
    
    # 预测
    predictions = []
    probabilities = []
    
    start_time = time.time()
    
    # 注意：Global-Local 模型不能直接使用 ForensicTTA，需要自己实现预测逻辑
    if is_global_local:
        print("⚠ Note: TTA is not fully supported for Global-Local models, using simplified prediction")
    
    for img_path in tqdm(all_images, desc="Predicting"):
        try:
            img = Image.open(img_path).convert('RGB')
            
            if is_global_local:
                # Global-Local 模型：需要同时准备 local patches 和 global image
                # 提取 local patches
                patches = patch_fn(img)  # (K, C, H, W)
                patches = transform_fn(patches)  # (K, C, H, W)
                patches = patches.unsqueeze(0).to(device)  # (1, K, C, H, W)
                
                # 准备 global image
                global_img = global_transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
                
                with torch.no_grad():
                    b, k, c, h, w = patches.shape
                    patches_flat = patches.view(b * k, c, h, w)  # (K, C, H, W)
                    
                    # 调用模型：传入两个参数
                    outputs = model(patches_flat, global_img)  # (K, 1)
                    outputs = outputs.view(b, k, 1)  # (1, K, 1)
                    probs = torch.sigmoid(outputs).mean(dim=1).squeeze()  # (1,) -> scalar
                    prob = probs.item()
            else:
                # 单流模型：可以使用 TTA
                if args.tta_mode != 'none':
                    # 使用 TTA
                    if 'predictor' not in locals():
                        from utils.tta import ForensicTTA
                        predictor = ForensicTTA(
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
                    prob = predictor.predict(img)
                else:
                    # 不使用 TTA，直接预测
                    patches = patch_fn(img)  # (K, C, H, W)
                    patches = transform_fn(patches)  # (K, C, H, W)
                    patches = patches.unsqueeze(0).to(device)  # (1, K, C, H, W)
                    
                    with torch.no_grad():
                        b, k, c, h, w = patches.shape
                        patches_flat = patches.view(b * k, c, h, w)  # (K, C, H, W)
                        outputs = model(patches_flat)  # (K, 1)
                        outputs = outputs.view(b, k, 1)  # (1, K, 1)
                        probs = torch.sigmoid(outputs).mean(dim=1).squeeze()  # (1,) -> scalar
                        prob = probs.item()
            
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
            probabilities.append(prob)
            
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            predictions.append(-1)  # 标记为错误
            probabilities.append(-1.0)
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    print("\n" + "="*70)
    print("预测统计")
    print("="*70)
    print(f"Total images: {len(all_images)}")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/len(all_images):.2f}s per image)")
    
    # 统计预测结果（排除错误的预测）
    valid_predictions = [p for p in predictions if p != -1]
    if len(valid_predictions) > 0:
        real_count = valid_predictions.count(0)
        fake_count = valid_predictions.count(1)
        
        print(f"\n预测结果分布:")
        print(f"  - Predicted as Real (0):      {real_count:5d} ({real_count/len(valid_predictions)*100:5.2f}%)")
        print(f"  - Predicted as AI-generated (1): {fake_count:5d} ({fake_count/len(valid_predictions)*100:5.2f}%)")
        
        # 概率统计
        valid_probs = [p for p in probabilities if p != -1.0]
        if len(valid_probs) > 0:
            print(f"\n概率统计:")
            print(f"  - Mean probability: {np.mean(valid_probs):.4f}")
            print(f"  - Median probability: {np.median(valid_probs):.4f}")
            print(f"  - Min probability: {np.min(valid_probs):.4f}")
            print(f"  - Max probability: {np.max(valid_probs):.4f}")
            print(f"  - Std probability: {np.std(valid_probs):.4f}")
    
    error_count = predictions.count(-1)
    if error_count > 0:
        print(f"\n⚠ Warning: {error_count} images failed to process")
    
    # 保存结果
    df = pd.DataFrame({
        'image_path': all_image_names,
        'predicted_label': predictions,
        'probability': probabilities,
        'predicted_class': ['Real' if p == 0 else 'Fake' if p == 1 else 'Error' for p in predictions]
    })
    
    df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results saved to: {args.output_file}")
    
    # 显示前10个预测结果
    print(f"\n前10个预测结果:")
    print(df.head(10).to_string(index=False))
    
    return df


def main():
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = load_model(args.model_path, device)
    
    results = predict_test_set(model, device)
    
    print("\n" + "="*70)
    print("✓ Prediction completed!")
    print("="*70)


if __name__ == '__main__':
    main()
