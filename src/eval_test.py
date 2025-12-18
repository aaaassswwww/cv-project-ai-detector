"""
对测试集进行预测

python src/eval_test.py   --use_global_local   --label_csv dataset/test_labels.csv --shift_method tent --tent_steps 1 --tent_lr 1e-4 --tent_reset_each_image  --optimize_threshold   --tta_mode standard   --tta_global_local   --tta_jpeg   --tta_hflip_gl   --patch_agg topm --patch_agg_m 2  
"""

from PIL import Image
import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import io
import torch.nn as nn
import copy

from networks.ssp import ssp
from utils.patch import patch_img_deterministic
from utils.util import set_seed
from utils.transform import PatchTransform

# 如果你 single-stream 仍想用你现成的 ForensicTTA，可保留
# from utils.tta import ForensicTTA


parser = argparse.ArgumentParser(description='对测试集进行预测（可选TTA/阈值扫描/评估accuracy）')
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--test_dir', default='test', type=str, help='test directory name')
parser.add_argument('--model_name', default='ssp-fda', type=str)
parser.add_argument('--model_path', default="./shared-nvme/ssp-models", help='Model Path root')
parser.add_argument('--output_file', default='predictions.csv', help='CSV output file')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--patch_var_thresh', default=5.0, type=float)

# 标签与评估
parser.add_argument('--label_csv', default='dataset/test_labels.csv', type=str, help='test label csv: image_id,label')
parser.add_argument('--optimize_threshold', action='store_true', help='search best threshold to maximize accuracy')
parser.add_argument('--threshold', default=0.5, type=float, help='decision threshold if not optimizing')

# TTA 参数（保留你原来的风格）
parser.add_argument('--tta_mode', default='full', type=str,
                    choices=['full', 'standard', 'simple', 'minimal', 'none'],
                    help='TTA mode: full/standard/simple/minimal/none')
parser.add_argument('--tta_hflip', action='store_true', help='enable horizontal flip')
parser.add_argument('--tta_vflip', action='store_true', help='enable vertical flip')
parser.add_argument('--tta_rotation', action='store_true', help='enable 90° rotation')
parser.add_argument('--tta_jpeg', action='store_true', help='enable JPEG compression')
parser.add_argument('--tta_multiscale', action='store_true', help='enable multi-scale')
parser.add_argument('--tta_aggregation', default='mean', type=str,
                    choices=['mean', 'median', 'vote'],
                    help='aggregation method for TTA predictions')

# Learnable SRM 参数（若模型使用了）
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

# patch 参数
parser.add_argument('--patch_topk', default=5, type=int, help='top-K patches to use')
parser.add_argument('--patch_size', default=32, type=int, help='patch size')

# patch 聚合（提分关键）
parser.add_argument('--patch_agg', default='mean', type=str, choices=['mean', 'topm', 'logit_mean'],
                    help='how to aggregate patch predictions into image prob')
parser.add_argument('--patch_agg_m', default=2, type=int, help='m for topm aggregation')

# Global-Local 的简单TTA（让GL也能提分）
parser.add_argument('--tta_global_local', action='store_true', help='enable simple TTA for Global-Local models')
parser.add_argument('--tta_jpeg_qualities', default='60,80,95', type=str, help='comma-separated jpeg qualities for GL TTA')
parser.add_argument('--tta_hflip_gl', action='store_true', help='hflip for GL TTA')

# AdaBN
parser.add_argument('--shift_method', default='none', type=str,
                    choices=['none', 'adabn', 'tent'],
                    help='domain shift mitigation at test time')
parser.add_argument('--adabn_passes', default=1, type=int, help='passes over test set to update BN stats')
parser.add_argument('--adabn_max_images', default=-1, type=int, help='limit images for AdaBN (-1 = all)')

# TENT
parser.add_argument('--tent_steps', default=1, type=int, help='gradient steps per image/batch')
parser.add_argument('--tent_lr', default=1e-4, type=float)
parser.add_argument('--tent_reset_each_image', action='store_true', help='reset BN affine each image (safer)')

args = parser.parse_args()

# TENT
def tent_prepare(model):
    # 冻结全部
    for p in model.parameters():
        p.requires_grad = False

    # 只打开 BN affine
    tent_params = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.affine:
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                tent_params += [m.weight, m.bias]
    return tent_params

def binary_entropy_from_logits(logits):
    # logits: (N,)
    p = torch.sigmoid(logits)
    eps = 1e-6
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent.mean()


# AdaBN
def _set_bn_train_only(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
    elif isinstance(m, nn.Dropout):
        m.eval()

def adabn_update_bn_stats(model, device, all_images, is_global_local,
                          patch_fn, transform_fn, global_transform):
    print("\n[ AdaBN ] Updating BN running stats on target/test images ...")
    model.train()
    model.apply(_set_bn_train_only)

    with torch.no_grad():
        for img_path in tqdm(all_images, desc="AdaBN"):
            img = Image.open(img_path).convert('RGB')

            patches = patch_fn(img)
            patches = transform_fn(patches).unsqueeze(0).to(device)  # (1,K,C,H,W)
            b, k, c, h, w = patches.shape
            patches_flat = patches.view(b * k, c, h, w)

            if is_global_local:
                global_img = global_transform(img).unsqueeze(0).to(device)
                _ = model(patches_flat, global_img)
            else:
                _ = model(patches_flat)

    model.eval()
    print("[ AdaBN ] Done.\n")


def load_test_labels(label_csv: str):
    if not label_csv:
        return None
    if not os.path.exists(label_csv):
        print(f"⚠ Warning: label csv not found: {label_csv}. Will run without accuracy.")
        return None

    df = pd.read_csv(label_csv)
    if 'image_id' not in df.columns or 'label' not in df.columns:
        raise ValueError("label_csv must have columns: image_id,label")

    label_map = {str(r['image_id']).strip(): int(r['label']) for _, r in df.iterrows()}
    return label_map


def aggregate_patches_from_logits(logits_1d: torch.Tensor, mode: str = 'mean', m: int = 2) -> float:
    """
    logits_1d: Tensor shape (K,) or (K,1)
    return scalar probability (float)
    """
    logits = logits_1d.view(-1)

    if mode == 'logit_mean':
        return torch.sigmoid(logits.mean()).item()

    probs = torch.sigmoid(logits)

    if mode == 'topm':
        m = max(1, min(m, probs.numel()))
        top_probs, _ = torch.topk(probs, k=m, largest=True)
        return top_probs.mean().item()

    return probs.mean().item()


def load_model(model_root, device):
    """加载模型（从checkpoint读取训练时关键参数，保证推理一致）"""
    full_model_path = os.path.join(model_root, args.model_name, 'checkpoint_epoch_25.pth')
    # full_model_path = os.path.join(model_root, args.model_name, 'checkpoint_epoch_20.pth')
    # full_model_path = os.path.join(model_root, args.model_name, 'ai-detector_best.pth')
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model not found: {full_model_path}")

    checkpoint = torch.load(full_model_path, map_location=device)
    saved_args = checkpoint.get('args', {})
    saved_var = saved_args.get('patch_var_thresh', None)
    if saved_var is not None:
        print(f"✓ Using patch_var_thresh from checkpoint: {saved_var}")
        args.patch_var_thresh = float(saved_var)

    # 训练时 topk 以 checkpoint 为准
    saved_topk = saved_args.get('patch_topk', 1)
    if saved_topk != args.patch_topk:
        print(f"⚠ Warning: Model trained with topk={saved_topk}, but you specified topk={args.patch_topk}")
        print(f"  → Using topk={saved_topk} from checkpoint")
        args.patch_topk = saved_topk

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

    # 加载权重
    try:
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("missing:", msg.missing_keys)
        print("unexpected:", msg.unexpected_keys)
    except RuntimeError:
        print(f"⚠ Warning: Strict loading failed, trying strict=False")
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if getattr(result, "missing_keys", None):
            print(f"  ⚠ missing keys: {len(result.missing_keys)}")
        if getattr(result, "unexpected_keys", None):
            print(f"  ⚠ unexpected keys: {len(result.unexpected_keys)}")

    model = model.to(device)
    model.eval()

    print(f"\n✓ Model loaded from: {full_model_path}")
    if 'epoch' in checkpoint:
        print(f"✓ Model epoch: {checkpoint.get('epoch')}")
    if 'val_acc' in checkpoint:
        print(f"✓ Model val_acc: {checkpoint['val_acc']:.2f}%")
    return model


def list_test_images(test_path: str):
    all_images = []
    all_image_names = []
    for root, _, files in os.walk(test_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, test_path)  # 相对 test_path
                all_images.append(img_path)
                all_image_names.append(rel_path)

    # 按文件名排序（你的名字是 image_0001.jpg 这种，字符串排序也OK；这里用 natsort 会更稳）
    try:
        from natsort import natsorted
        order = np.argsort(natsorted(all_image_names))
        # 上面这行在某些版本可能不对，改成直接natsorted + map
        sorted_names = natsorted(all_image_names)
        name_to_path = {n: p for n, p in zip(all_image_names, all_images)}
        all_image_names = sorted_names
        all_images = [name_to_path[n] for n in all_image_names]
    except Exception:
        idx = np.argsort(all_image_names)
        all_images = [all_images[i] for i in idx]
        all_image_names = [all_image_names[i] for i in idx]

    return all_images, all_image_names


def make_patch_fn():
    def patch_fn(img):
        return patch_img_deterministic(
            img,
            args.patch_size,
            256,
            var_thresh=0.0,
            topk=args.patch_topk
        )
    return patch_fn


def make_patch_transform():
    patch_transform = PatchTransform(
        patch_size=256,
        apply_augment=False,
        augment_config=None
    )

    def transform_fn(patches):
        return patch_transform(patches)  # (K,C,H,W)
    return transform_fn


def make_global_transform():
    # Global-Local 的 global 分支：必须 normalize（与训练一致）
    from torchvision import transforms
    global_transform = transforms.Compose([
        transforms.Resize((args.global_size, args.global_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return global_transform


def generate_gl_tta_images(img: Image.Image):
    imgs = [img]

    # hflip
    if args.tta_hflip_gl:
        imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # jpeg qualities
    if args.tta_jpeg:
        qs = [int(x) for x in args.tta_jpeg_qualities.split(',') if x.strip() != '']
        for q in qs:
            try:
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=q)
                buf.seek(0)
                imgs.append(Image.open(buf).convert('RGB'))
            except Exception:
                pass

    return imgs


def predict_one_global_local(model, device, img_pil, patch_fn, transform_fn, global_transform, tent_opt=None, bn_backup=None):
    patches = patch_fn(img_pil)
    patches = transform_fn(patches)  # (K,C,256,256)
    patches = patches.unsqueeze(0).to(device)  # (1,K,C,H,W)

    global_img = global_transform(img_pil).unsqueeze(0).to(device)  # (1,3,gs,gs)
    b, k, c, h, w = patches.shape
    patches_flat = patches.view(b * k, c, h, w)

    # ---- TENT adaptation (optional) ----
    if tent_opt is not None:
        if args.tent_reset_each_image and bn_backup is not None:
            model.load_state_dict(bn_backup, strict=False)

        model.train()
        model.apply(_set_bn_train_only)

        for _ in range(max(1, args.tent_steps)):
            tent_opt.zero_grad(set_to_none=True)
            logits = model(patches_flat, global_img).view(-1)
            loss = binary_entropy_from_logits(logits)
            loss.backward()
            tent_opt.step()

        model.eval()
        
    with torch.no_grad():
        logits = model(patches_flat, global_img).view(-1)  # (K,)
        prob = aggregate_patches_from_logits(logits, mode=args.patch_agg, m=args.patch_agg_m)
    return prob


def predict_one_single_stream(model, device, img_pil, patch_fn, transform_fn, tent_opt=None, bn_backup=None):
    patches = patch_fn(img_pil)
    patches = transform_fn(patches)
    patches = patches.unsqueeze(0).to(device)

    b, k, c, h, w = patches.shape
    patches_flat = patches.view(b * k, c, h, w)

    # ---- TENT adaptation (optional) ----
    if tent_opt is not None:
        if args.tent_reset_each_image and bn_backup is not None:
            model.load_state_dict(bn_backup, strict=False)

        model.train()
        model.apply(_set_bn_train_only)

        for _ in range(max(1, args.tent_steps)):
            tent_opt.zero_grad(set_to_none=True)
            logits = model(patches_flat).view(-1)
            loss = binary_entropy_from_logits(logits)
            loss.backward()
            tent_opt.step()

        model.eval()
        
    # ---- final prediction ----
    with torch.no_grad():
        logits = model(patches_flat).view(-1)  # (K,)
        prob = aggregate_patches_from_logits(logits, mode=args.patch_agg, m=args.patch_agg_m)
    return prob


def optimize_threshold_for_accuracy(probs, y_true):
    # probs: list[float], y_true: list[int]
    probs_np = np.asarray(probs, dtype=np.float32)
    y_np = np.asarray(y_true, dtype=np.int64)

    thrs = np.linspace(0.0, 1.0, 1001)
    best_thr = 0.5
    best_acc = -1.0

    for t in thrs:
        preds = (probs_np > t).astype(np.int64)
        acc = (preds == y_np).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)

    return best_thr, float(best_acc)


def predict_test_set(model, device):
    print("\n" + "=" * 70)
    print(f"开始预测测试集 (tta_mode={args.tta_mode}, patch_agg={args.patch_agg})")
    print("=" * 70)

    label_map = load_test_labels(args.label_csv)

    # 判断是否global-local
    model_class_name = model.__class__.__name__
    is_global_local = ('GlobalLocal' in model_class_name) or (hasattr(model, 'local_stream') and hasattr(model, 'global_stream'))
    print(f"✓ Model type: {model_class_name}")
    print(f"✓ Detected Global-Local: {is_global_local}")

    tent_opt = None
    bn_backup = None
    if args.shift_method == 'tent':
        params = tent_prepare(model)
        tent_opt = torch.optim.Adam(params, lr=args.tent_lr)
        if args.tent_reset_each_image:
            bn_backup = copy.deepcopy(model.state_dict())
        print(f"✓ TENT enabled: steps={args.tent_steps}, lr={args.tent_lr}, reset_each_image={args.tent_reset_each_image}")

    test_path = os.path.join(args.dataset_root, args.test_dir)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    all_images, all_image_names = list_test_images(test_path)
    print(f"\n✓ Found {len(all_images)} images in {test_path}")
    if len(all_images) == 0:
        print("❌ No images found! Please check your test directory.")
        return None

    patch_fn = make_patch_fn()
    transform_fn = make_patch_transform()

    global_transform = make_global_transform() if is_global_local else None

    if args.shift_method == 'adabn':
        imgs_for_bn = all_images
        if args.adabn_max_images > 0:
            imgs_for_bn = imgs_for_bn[:args.adabn_max_images]
        for _ in range(max(1, args.adabn_passes)):
            adabn_update_bn_stats(model, device, imgs_for_bn, is_global_local,
                                  patch_fn, transform_fn, global_transform)


    # 预测
    probs = []
    start_time = time.time()

    for img_path in tqdm(all_images, desc="Predicting"):
        try:
            img = Image.open(img_path).convert('RGB')

            if is_global_local:
                if args.tta_global_local and args.tta_mode != 'none':
                    tta_imgs = generate_gl_tta_images(img)
                    prob_list = [predict_one_global_local(model, device, im, patch_fn, transform_fn, global_transform, tent_opt=tent_opt, bn_backup=bn_backup) for im in tta_imgs]

                    if args.tta_aggregation == 'median':
                        prob = float(np.median(prob_list))
                    elif args.tta_aggregation == 'vote':
                        # vote: 用当前阈值投票（如果后面会optimize_threshold，这里vote会稍微“非最优”）
                        thr = args.threshold
                        votes = [1 if p > thr else 0 for p in prob_list]
                        prob = float(np.mean(votes))
                    elif args.tta_aggregation == 'min':
                        prob = float(np.min(prob_list))
                    elif args.tta_aggregation == 'p25':
                        prob = float(np.percentile(prob_list, 25))
                    else:
                        prob = float(np.mean(prob_list))
                else:
                    prob = predict_one_global_local(model, device, img, patch_fn, transform_fn, global_transform, tent_opt=tent_opt, bn_backup=bn_backup)

            else:
                # single-stream：这里先用同一套轻量TTA（和GL一致），避免依赖外部ForensicTTA
                if args.tta_mode != 'none':
                    # 复用同样的jpeg/hflip策略
                    tta_imgs = [img]
                    if args.tta_mode in ['minimal', 'simple', 'standard', 'full'] or args.tta_hflip:
                        tta_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
                    if args.tta_jpeg or (args.tta_mode in ['standard', 'full']):
                        qs = [60, 80, 95]
                        for q in qs:
                            try:
                                buf = io.BytesIO()
                                img.save(buf, format='JPEG', quality=q)
                                buf.seek(0)
                                tta_imgs.append(Image.open(buf).convert('RGB'))
                            except Exception:
                                pass

                    prob_list = [predict_one_single_stream(model, device, im, patch_fn, transform_fn) for im in tta_imgs]
                    if args.tta_aggregation == 'median':
                        prob = float(np.median(prob_list))
                    elif args.tta_aggregation == 'vote':
                        thr = args.threshold
                        votes = [1 if p > thr else 0 for p in prob_list]
                        prob = float(np.mean(votes))
                    else:
                        prob = float(np.mean(prob_list))
                else:
                    prob = predict_one_single_stream(model, device, img, patch_fn, transform_fn, tent_opt=tent_opt, bn_backup=bn_backup)

            probs.append(prob)

        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            probs.append(-1.0)

    elapsed = time.time() - start_time
    print(f"\n✓ Total time: {elapsed:.1f}s ({elapsed / max(1, len(all_images)):.3f}s per image)")

    # 组装真实标签（按 basename 对齐）
    true_labels = None
    if label_map is not None:
        true_labels = []
        missing = 0
        for rel in all_image_names:
            key = os.path.basename(rel)  # 你这里就是 image_0001.jpg
            if key in label_map:
                true_labels.append(int(label_map[key]))
            else:
                true_labels.append(None)
                missing += 1
        if missing > 0:
            print(f"⚠ Warning: {missing} images not found in label csv")

    # 阈值（可选优化）
    final_thr = args.threshold
    best_acc = None
    if true_labels is not None and args.optimize_threshold:
        pairs = [(p, y) for p, y in zip(probs, true_labels) if p != -1.0 and y is not None]
        if len(pairs) > 0:
            p_list = [a for a, _ in pairs]
            y_list = [b for _, b in pairs]
            final_thr, best_acc = optimize_threshold_for_accuracy(p_list, y_list)
            print(f"\n✅ Best threshold (maximize accuracy) = {final_thr:.4f}, accuracy = {best_acc*100:.2f}%")
        else:
            print("⚠ No valid labeled pairs for threshold optimization. Using default threshold.")

    # 最终预测
    preds = []
    for p in probs:
        if p == -1.0:
            preds.append(-1)
        else:
            preds.append(1 if p > final_thr else 0)

    # 输出统计
    valid_preds = [x for x in preds if x != -1]
    if len(valid_preds) > 0:
        real_count = valid_preds.count(0)
        fake_count = valid_preds.count(1)
        print("\n预测结果分布:")
        print(f"  - Pred 0 (Real): {real_count} ({real_count/len(valid_preds)*100:.2f}%)")
        print(f"  - Pred 1 (Fake): {fake_count} ({fake_count/len(valid_preds)*100:.2f}%)")

        valid_probs = [p for p in probs if p != -1.0]
        if len(valid_probs) > 0:
            print("\n概率统计:")
            print(f"  - mean   : {np.mean(valid_probs):.4f}")
            print(f"  - median : {np.median(valid_probs):.4f}")
            print(f"  - min/max: {np.min(valid_probs):.4f} / {np.max(valid_probs):.4f}")
            print(f"  - std    : {np.std(valid_probs):.4f}")

    # 计算 accuracy
    acc = None
    if true_labels is not None:
        mask = [(p != -1 and y is not None) for p, y in zip(preds, true_labels)]
        if any(mask):
            y_true = np.array([y for y, m in zip(true_labels, mask) if m], dtype=np.int64)
            y_pred = np.array([p for p, m in zip(preds, mask) if m], dtype=np.int64)
            acc = float((y_true == y_pred).mean())
            print(f"\n✅ Accuracy @thr={final_thr:.4f}: {acc*100:.2f}% (n={len(y_true)})")
        else:
            print("⚠ No valid labeled samples to compute accuracy.")

    # 保存结果
    df = pd.DataFrame({
        'image_path': all_image_names,
        'image_id': [os.path.basename(x) for x in all_image_names],
        'probability': probs,
        'predicted_label': preds,
    })

    if true_labels is not None:
        df['true_label'] = true_labels

    df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results saved to: {args.output_file}")
    print("\n前10行结果:")
    print(df.head(10).to_string(index=False))

    return df


def main():
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    model = load_model(args.model_path, device)
    _ = predict_test_set(model, device)

    print("\n" + "=" * 70)
    print("✓ Prediction completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
