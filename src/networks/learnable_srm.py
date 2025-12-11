"""
可学习的 SRM（Steganalysis Rich Model）高通滤波层

原理：
- 经典 SRM 是一组手工设计的高通滤波核，用于强调图像的高频残差信号
- 本实现将 SRM 转为可学习卷积层，保留高频初始化作为先验
- 允许网络在训练中微调以适应数据集分布，提升对伪造痕迹的敏感性

特点：
- 参数量极少（通常 < 1K）
- 可以 freeze/unfreeze 进行阶段性训练
- 可单独调整学习率以控制过拟合
- 完全兼容现有 pipeline
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# 经典 SRM 核心库（3x3 高通滤波器）
SRM_KERNELS_3x3 = np.array([
    # High-pass filter 1
    [[-2,  -1,   0],
     [-1,   4,  -1],
     [ 0,  -1,  -2]],
    
    # High-pass filter 2
    [[0,  -1,   0],
     [-1,  4,  -1],
     [0,  -1,   0]],
    
    # High-pass filter 3
    [[1,  -2,   1],
     [-2,  4,  -2],
     [1,  -2,   1]],
    
    # High-pass filter 4 (diagonal)
    [[-1,  2,  -2],
     [2,  -4,   2],
     [-2,  2,  -1]],
    
    # High-pass filter 5
    [[-2,  -1,   0],
     [-1,   4,  -1],
     [0,  -1,  -2]],
    
    # High-pass filter 6
    [[0,   -1,   0],
     [-1,   5,  -1],
     [0,  -1,   0]],
], dtype=np.float32)

# 扩展到 5x5 的 SRM 核（常见应用）
SRM_KERNELS_5x5 = np.array([
    # Filter 1: LoG-like
    [[0,   0,  -1,   0,   0],
     [0,  -1,  -2,  -1,   0],
     [-1, -2,  16,  -2,  -1],
     [0,  -1,  -2,  -1,   0],
     [0,   0,  -1,   0,   0]],
    
    # Filter 2: Prewitt horizontal
    [[-1,  -1,  -1,  -1,  -1],
     [0,   0,   0,   0,   0],
     [1,   1,   1,   1,   1],
     [0,   0,   0,   0,   0],
     [1,   1,   1,   1,   1]],
    
    # Filter 3: Prewitt vertical
    [[-1,   0,   1,   0,   1],
     [-1,   0,   1,   0,   1],
     [-1,   0,   1,   0,   1],
     [-1,   0,   1,   0,   1],
     [-1,   0,   1,   0,   1]],
    
    # Filter 4: High-pass diagonal
    [[1,  -2,   0,  -2,   1],
     [-2,  4,   0,   4,  -2],
     [0,   0,  -8,   0,   0],
     [-2,  4,   0,   4,  -2],
     [1,  -2,   0,  -2,   1]],
    
    # Filter 5: Another high-pass
    [[0,   0,  -2,   0,   0],
     [0,  -2,  -4,  -2,   0],
     [-2, -4,  24,  -4,  -2],
     [0,  -2,  -4,  -2,   0],
     [0,   0,  -2,   0,   0]],
    
    # Filter 6: Edge detector
    [[-1,  -1,  -1,  -1,  -1],
     [-1,  -1,  -1,  -1,  -1],
     [-1,  -1,  32,  -1,  -1],
     [-1,  -1,  -1,  -1,  -1],
     [-1,  -1,  -1,  -1,  -1]],
], dtype=np.float32)


def create_srm_kernels(kernel_size: int = 5, num_kernels: int = 12) -> torch.Tensor:
    """
    创建 SRM kernel 张量
    
    参数：
        kernel_size: 3 或 5
        num_kernels: 输出 channel 数，应该 <= 基础核数
    
    返回：
        shape (num_kernels, 3, kernel_size, kernel_size) 的张量
        - num_kernels: 输出通道数
        - 3: RGB 输入通道
        - kernel_size x kernel_size: 卷积核大小
    """
    if kernel_size == 3:
        srm_base = SRM_KERNELS_3x3
    elif kernel_size == 5:
        srm_base = SRM_KERNELS_5x5
    else:
        raise ValueError(f"kernel_size must be 3 or 5, got {kernel_size}")
    
    # srm_base shape: (base_num, kernel_size, kernel_size)
    base_num = srm_base.shape[0]
    
    if num_kernels > base_num:
        print(f"⚠ Warning: num_kernels={num_kernels} > base kernels={base_num}")
        print(f"  Will tile and perturb kernels to reach {num_kernels}")
    
    # 创建输出 kernels
    kernels_list = []
    
    # 使用基础核
    for i in range(min(num_kernels, base_num)):
        kernel = srm_base[i:i+1]  # (1, k, k)
        # 复制到 3 个 RGB 通道，权重相同（因为 SRM 通常是灰度）
        kernel_3ch = np.tile(kernel, (3, 1, 1))  # (3, k, k)
        kernels_list.append(kernel_3ch)
    
    # 如果需要更多核，对基础核进行微小扰动
    if num_kernels > base_num:
        np.random.seed(42)  # 保证可重复性
        for i in range(num_kernels - base_num):
            base_idx = i % base_num
            kernel = srm_base[base_idx:base_idx+1].copy()
            # 加入微小噪声扰动
            noise = np.random.randn(*kernel.shape) * 0.05
            kernel = kernel + noise
            kernel_3ch = np.tile(kernel, (3, 1, 1))
            kernels_list.append(kernel_3ch)
    
    # 合并成 (num_kernels, 3, k, k)
    kernels = np.stack(kernels_list, axis=0)
    
    # 归一化到 [-1, 1] 范围（保持 SRM 的特性）
    kernels = kernels / (np.abs(kernels).max() + 1e-8)
    
    return torch.from_numpy(kernels).float()


class LearnableSRM(nn.Module):
    """
    可学习的 SRM 高通滤波层
    
    参数：
        in_channels: 输入通道数（通常为 3）
        out_channels: 输出通道数（默认 12）
        kernel_size: 卷积核大小（3 或 5，默认 5）
        use_bias: 是否使用偏置（推荐 False，因为 SRM 是残差滤波）
        freeze_init_epochs: 冻结该层多少个 epoch（0 表示不冻结）
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 12,
        kernel_size: int = 5,
        use_bias: bool = False,
        freeze_init_epochs: int = 0,
    ):
        super().__init__()
        
        assert kernel_size in [3, 5], "kernel_size must be 3 or 5"
        assert out_channels > 0, "out_channels must > 0"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.freeze_init_epochs = freeze_init_epochs
        
        # 创建卷积层
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=use_bias,
        )
        
        # 用 SRM kernel 初始化权重
        srm_kernels = create_srm_kernels(kernel_size, out_channels)
        # srm_kernels: (out_ch, 3, k, k)
        
        # 如果 in_channels != 3，进行调整
        if in_channels != 3:
            # 简单处理：如果输入通道不是 3，复制或平均 SRM kernels
            if in_channels == 1:
                # 灰度图：平均 RGB
                srm_kernels = srm_kernels.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                # 更多通道：重复或扩展
                srm_kernels = srm_kernels.repeat(1, (in_channels // 3) + 1, 1, 1)
                srm_kernels = srm_kernels[:, :in_channels, :, :]
        
        # 复制到权重
        self.conv.weight.data.copy_(srm_kernels)
        
        print(f"✓ LearnableSRM initialized:")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Output channels: {out_channels}")
        print(f"  - Kernel size: {kernel_size}x{kernel_size}")
        print(f"  - Parameters: {out_channels * in_channels * kernel_size * kernel_size}")
        print(f"  - Freeze init epochs: {freeze_init_epochs}")
    
    def freeze(self):
        """冻结该层的参数"""
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
    
    def unfreeze(self):
        """解冻该层的参数"""
        self.conv.weight.requires_grad = True
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = True
    
    def set_learning_rate_scale(self, scale: float):
        """
        设置该层的学习率缩放因子
        
        在优化器中使用 param_groups 实现：
        Example:
            for param_group in optimizer.param_groups:
                if 'srm' in param_group.get('name', ''):
                    param_group['lr'] = base_lr * scale
        """
        self._lr_scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, C, H, W)
        输出: (B, out_channels, H, W)
        """
        return self.conv(x)
    
    def __repr__(self):
        return (
            f"LearnableSRM(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}x{self.kernel_size}, "
            f"freeze_init_epochs={self.freeze_init_epochs})"
        )


class SRMBlock(nn.Module):
    """
    SRM 块：包含可学习 SRM + 可选的 BN 和激活
    
    通常配置：
    - SRM + BatchNorm + ReLU: 作为特征提取
    - SRM 仅输出: 作为线性残差提取器
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 12,
        kernel_size: int = 5,
        use_bn: bool = True,
        activation: Optional[str] = None,
        freeze_init_epochs: int = 0,
    ):
        super().__init__()
        
        self.srm = LearnableSRM(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=not use_bn,  # 如果有 BN 就不要偏置
            freeze_init_epochs=freeze_init_epochs,
        )
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.01, inplace=True)
        elif activation is None:
            self.act = None
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
    def freeze(self):
        self.srm.freeze()
        if self.bn is not None:
            for param in self.bn.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        self.srm.unfreeze()
        if self.bn is not None:
            for param in self.bn.parameters():
                param.requires_grad = True


def create_learnable_srm(
    in_channels: int = 3,
    out_channels: int = 12,
    kernel_size: int = 5,
    block_type: str = 'srm_only',
    freeze_init_epochs: int = 0,
) -> nn.Module:
    """
    工厂函数：创建 SRM 模块
    
    参数：
        block_type: 
            'srm_only': 仅 SRM，无激活
            'srm_bn': SRM + BatchNorm
            'srm_bn_relu': SRM + BatchNorm + ReLU
            'srm_bn_leaky': SRM + BatchNorm + LeakyReLU
    """
    if block_type == 'srm_only':
        return LearnableSRM(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            freeze_init_epochs=freeze_init_epochs,
        )
    elif block_type == 'srm_bn':
        return SRMBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bn=True,
            activation=None,
            freeze_init_epochs=freeze_init_epochs,
        )
    elif block_type == 'srm_bn_relu':
        return SRMBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bn=True,
            activation='relu',
            freeze_init_epochs=freeze_init_epochs,
        )
    elif block_type == 'srm_bn_leaky':
        return SRMBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bn=True,
            activation='leaky_relu',
            freeze_init_epochs=freeze_init_epochs,
        )
    else:
        raise ValueError(f"Unknown block_type: {block_type}")
