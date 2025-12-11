"""
自定义 Transform 类，用于处理 Top-K patch 的增强流程
避免使用过多的 Lambda 表达式，提高代码可读性和可维护性
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import List, Union, Callable, Optional
from .augment import (
    RandomFreqPerturbation,
    RandomGaussianBlurProb,
    RandomGaussianNoise,
    RandomJPEGCompression,
    RandomResample,
)


class PatchTransform:
    """
    处理单个或多个 patch 的统一 Transform 类
    
    功能：
    1. 自动处理单 patch 或多 patch (list) 输入
    2. 统一应用 resize、augmentation、normalization
    3. 最终输出为 (K, C, H, W) 的 torch.Tensor，K=1 或 K>1
    
    参数：
        patch_size: patch 目标尺寸 (默认 256)
        apply_augment: 是否应用数据增强 (训练时 True，验证时 False)
        augment_config: 数据增强配置字典
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        apply_augment: bool = True,
        augment_config: Optional[dict] = None
    ):
        self.patch_size = patch_size
        self.apply_augment = apply_augment
        
        # 默认增强配置
        default_config = {
            'jpeg_p_patch': 0.05,
            'jpeg_quality_min': 30,
            'jpeg_quality_max': 95,
            'blur_p': 0.15,
            'blur_kernel_size': 3,
            'blur_sigma_min': 0.1,
            'blur_sigma_max': 2.0,
            'resample_p': 0.15,
            'resample_scale_min': 0.5,
            'resample_scale_max': 0.9,
            'noise_p': 0.1,
            'noise_sigma_min': 0.005,
            'noise_sigma_max': 0.02,
            'freq_p': 0.1,
            'freq_scale_min': 0.7,
            'freq_scale_max': 1.3,
            'freq_radius': 0.25,
        }
        
        self.config = augment_config if augment_config else default_config
        
        # 初始化增强操作
        if self.apply_augment:
            self.jpeg_aug = RandomJPEGCompression(
                p=self.config['jpeg_p_patch'],
                quality_range=(self.config['jpeg_quality_min'], self.config['jpeg_quality_max']),
            )
            self.blur_aug = RandomGaussianBlurProb(
                p=self.config['blur_p'],
                kernel_size=self.config['blur_kernel_size'],
                sigma_range=(self.config['blur_sigma_min'], self.config['blur_sigma_max']),
            )
            self.resample_aug = RandomResample(
                p=self.config['resample_p'],
                scale_range=(self.config['resample_scale_min'], self.config['resample_scale_max']),
            )
            self.noise_aug = RandomGaussianNoise(
                p=self.config['noise_p'],
                sigma_range=(self.config['noise_sigma_min'], self.config['noise_sigma_max']),
            )
            self.freq_aug = RandomFreqPerturbation(
                p=self.config['freq_p'],
                scale_range=(self.config['freq_scale_min'], self.config['freq_scale_max']),
                radius=self.config['freq_radius'],
            )
        
        # 基础操作（不使用 ImageNet normalize，forensics 任务不需要）
        self.resize = transforms.Resize((patch_size, patch_size))
        self.to_tensor = transforms.ToTensor()  # 转为 [0,1] 范围
        # 注意：不使用 normalize，因为：
        # 1. normalize 会放大噪声增强的影响 (noise/0.22 ≈ 4.5x)
        # 2. SRM 高频特征不应被 normalize 扭曲
        # 3. Forensics 任务通常不需要 ImageNet 统计量
    
    def _ensure_list(self, patches: Union[Image.Image, List[Image.Image]]) -> List[Image.Image]:
        """确保输入为列表格式"""
        if isinstance(patches, list):
            return patches
        else:
            return [patches]
    
    def _apply_pil_transform(self, patch: Image.Image) -> Image.Image:
        """对单个 PIL Image 应用 PIL 级别的增强"""
        # 问题7修复：只在patch尺寸不匹配时才resize，避免双重resize
        if patch.size != (self.patch_size, self.patch_size):
            patch = self.resize(patch)
        
        if self.apply_augment:
            # 应用 JPEG 压缩增强（在 patch 之后）
            patch = self.jpeg_aug(patch)
            # 应用高斯模糊
            patch = self.blur_aug(patch)
            # 应用重采样
            patch = self.resample_aug(patch)
        
        return patch
    
    def _apply_tensor_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """对单个 Tensor 应用 Tensor 级别的增强"""
        if self.apply_augment:
            # 应用高斯噪声（在 [0,1] 范围内）
            tensor = self.noise_aug(tensor)
            # 应用频率扰动（在 [0,1] 范围内）
            tensor = self.freq_aug(tensor)
        
        # 不再使用 normalize，保持 [0,1] 范围，避免破坏 SRM 特征
        return tensor
    
    def __call__(self, patches: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        对输入的 patch(es) 应用完整的转换流程
        
        输入:
            patches: 单个 PIL.Image 或 PIL.Image 列表
        
        输出:
            torch.Tensor of shape (K, C, H, W)
            - K=1 when input is single patch
            - K>1 when input is list of patches
        """
        # 1. 确保为列表格式
        patch_list = self._ensure_list(patches)
        
        # 2. 应用 PIL 级别的转换（resize + PIL augmentations）
        pil_transformed = [self._apply_pil_transform(p) for p in patch_list]
        
        # 3. 转换为 Tensor
        tensors = [self.to_tensor(p) for p in pil_transformed]
        
        # 4. 应用 Tensor 级别的增强和标准化
        final_tensors = [self._apply_tensor_transform(t) for t in tensors]
        
        # 5. Stack 成 (K, C, H, W)
        output = torch.stack(final_tensors, dim=0)
        
        return output
    
    def __repr__(self):
        aug_status = "with augmentation" if self.apply_augment else "without augmentation"
        return f"PatchTransform(patch_size={self.patch_size}, {aug_status})"


class GlobalJPEGTransform:
    """
    全局 JPEG 压缩增强（在提取 patch 之前应用）
    独立出来是因为它作用于完整图像而非 patch
    """
    
    def __init__(self, p: float = 0.2, quality_range: tuple = (30, 95)):
        self.jpeg_aug = RandomJPEGCompression(p=p, quality_range=quality_range)
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return self.jpeg_aug(img)
    
    def __repr__(self):
        return f"GlobalJPEGTransform(p={self.jpeg_aug.p}, quality={self.jpeg_aug.quality_range})"


def build_train_transform(args, patch_fn: Callable) -> transforms.Compose:
    """
    构建训练时的 Transform pipeline
    
    参数：
        args: 命令行参数对象
        patch_fn: 提取 patch 的函数（已经配置好 topk、var_thresh 等参数）
    
    返回：
        transforms.Compose 对象
    """
    augment_config = {
        'jpeg_p_patch': args.jpeg_p_patch,
        'jpeg_quality_min': args.jpeg_quality_min,
        'jpeg_quality_max': args.jpeg_quality_max,
        'blur_p': args.blur_p,
        'blur_kernel_size': args.blur_kernel_size,
        'blur_sigma_min': args.blur_sigma_min,
        'blur_sigma_max': args.blur_sigma_max,
        'resample_p': args.resample_p,
        'resample_scale_min': args.resample_scale_min,
        'resample_scale_max': args.resample_scale_max,
        'noise_p': args.noise_p,
        'noise_sigma_min': args.noise_sigma_min,
        'noise_sigma_max': args.noise_sigma_max,
        'freq_p': args.freq_p,
        'freq_scale_min': args.freq_scale_min,
        'freq_scale_max': args.freq_scale_max,
        'freq_radius': args.freq_radius,
    }
    
    return transforms.Compose([
        # 1. 全局 JPEG 压缩（作用于原始图像）
        GlobalJPEGTransform(
            p=args.jpeg_p_global,
            quality_range=(args.jpeg_quality_min, args.jpeg_quality_max),
        ),
        # 2. 提取 patch（返回单个或多个 patch）
        transforms.Lambda(patch_fn),
        # 3. 统一的 patch 处理（resize + augment + normalize）
        PatchTransform(
            patch_size=256,
            apply_augment=True,
            augment_config=augment_config,
        ),
    ])


def build_val_transform(args, patch_fn: Callable) -> transforms.Compose:
    """
    构建验证/测试时的 Transform pipeline
    
    参数：
        args: 命令行参数对象
        patch_fn: 提取 patch 的函数（deterministic 模式）
    
    返回：
        transforms.Compose 对象
    """
    return transforms.Compose([
        # 1. 提取 patch（deterministic 模式）
        transforms.Lambda(patch_fn),
        # 2. 统一的 patch 处理（仅 resize + normalize，无增强）
        PatchTransform(
            patch_size=256,
            apply_augment=False,
            augment_config=None,
        ),
    ])


class DualStreamTransform:
    """
    Global-Local Dual Stream 专用 Transform
    
    同时输出：
    1. Global: 完整的原图（resize 到固定尺寸）
    2. Local: 提取的 patch（经过增强处理）
    
    这样 Global Stream 才能真正看到全局布局和语义信息
    """
    
    def __init__(
        self,
        global_size: int,
        patch_fn: Callable,
        patch_size: int = 256,
        apply_augment: bool = True,
        augment_config: Optional[dict] = None,
        apply_global_jpeg: bool = True,
        jpeg_p: float = 0.2,
        jpeg_quality_range: tuple = (30, 95),
    ):
        """
        参数：
            global_size: 全局图的目标尺寸（例如 384）
            patch_fn: patch 提取函数
            patch_size: patch 的目标尺寸（例如 256）
            apply_augment: 是否对 patch 应用增强
            augment_config: patch 增强配置
            apply_global_jpeg: 是否在提取 patch 前对原图应用 JPEG 压缩
            jpeg_p: JPEG 压缩概率
            jpeg_quality_range: JPEG 质量范围
        """
        self.global_size = global_size
        self.patch_fn = patch_fn
        self.apply_global_jpeg = apply_global_jpeg
        
        # 全局 JPEG 增强（可选）
        if apply_global_jpeg:
            self.global_jpeg = RandomJPEGCompression(p=jpeg_p, quality_range=jpeg_quality_range)
        
        # 全局图处理：Resize + ToTensor
        # 注意：保持 [0,1] 范围，不使用 ImageNet normalize
        self.global_transform = transforms.Compose([
            transforms.Resize((global_size, global_size)),
            transforms.ToTensor(),
        ])
        
        # Local patch 处理
        self.patch_transform = PatchTransform(
            patch_size=patch_size,
            apply_augment=apply_augment,
            augment_config=augment_config,
        )
    
    def __call__(self, img: Image.Image) -> dict:
        """
        输入：PIL Image（原始图像）
        输出：字典 {'global': Tensor(C,H,W), 'local': Tensor(K,C,H,W)}
        """
        # 1. 可选的全局 JPEG 压缩（在分流前应用）
        if self.apply_global_jpeg:
            img = self.global_jpeg(img)
        
        # 2. Global Stream: 处理完整图像
        x_global = self.global_transform(img)  # (3, global_size, global_size)
        
        # 3. Local Stream: 提取 patch 并处理
        patches = self.patch_fn(img)  # 返回 PIL Image 或 list of PIL Images
        x_local = self.patch_transform(patches)  # (K, 3, 256, 256)
        
        return {
            'global': x_global,
            'local': x_local,
        }
    
    def __repr__(self):
        return (f"DualStreamTransform(\n"
                f"  global_size={self.global_size},\n"
                f"  patch_transform={self.patch_transform}\n"
                f")")


def build_dual_stream_train_transform(args, patch_fn: Callable) -> DualStreamTransform:
    """
    构建 Global-Local Dual Stream 训练的 Transform
    
    参数：
        args: 命令行参数对象（需要包含 global_size）
        patch_fn: patch 提取函数
    
    返回：
        DualStreamTransform 对象
    """
    augment_config = {
        'jpeg_p_patch': args.jpeg_p_patch,
        'jpeg_quality_min': args.jpeg_quality_min,
        'jpeg_quality_max': args.jpeg_quality_max,
        'blur_p': args.blur_p,
        'blur_kernel_size': args.blur_kernel_size,
        'blur_sigma_min': args.blur_sigma_min,
        'blur_sigma_max': args.blur_sigma_max,
        'resample_p': args.resample_p,
        'resample_scale_min': args.resample_scale_min,
        'resample_scale_max': args.resample_scale_max,
        'noise_p': args.noise_p,
        'noise_sigma_min': args.noise_sigma_min,
        'noise_sigma_max': args.noise_sigma_max,
        'freq_p': args.freq_p,
        'freq_scale_min': args.freq_scale_min,
        'freq_scale_max': args.freq_scale_max,
        'freq_radius': args.freq_radius,
    }
    
    return DualStreamTransform(
        global_size=getattr(args, 'global_size', 384),  # 默认 384
        patch_fn=patch_fn,
        patch_size=256,
        apply_augment=True,
        augment_config=augment_config,
        apply_global_jpeg=True,
        jpeg_p=args.jpeg_p_global,
        jpeg_quality_range=(args.jpeg_quality_min, args.jpeg_quality_max),
    )


def build_dual_stream_val_transform(args, patch_fn: Callable) -> DualStreamTransform:
    """
    构建 Global-Local Dual Stream 验证/测试的 Transform
    
    参数：
        args: 命令行参数对象
        patch_fn: patch 提取函数（deterministic 模式）
    
    返回：
        DualStreamTransform 对象
    """
    return DualStreamTransform(
        global_size=getattr(args, 'global_size', 384),
        patch_fn=patch_fn,
        patch_size=256,
        apply_augment=False,  # 验证时不增强
        augment_config=None,
        apply_global_jpeg=False,  # 验证时不压缩
    )
