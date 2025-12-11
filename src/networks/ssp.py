import torch
from torch import nn
from networks.resnet import resnet50
from networks.srm_conv import SRMConv2d_simple
from networks.learnable_srm import create_learnable_srm
import torch.nn.functional as F
from typing import Optional


class ssp(nn.Module):
    """
    SSP (Single Simple Patch) 模型
    
    参数：
        pretrain: 是否使用 ResNet50 的预训练权重
        topk: 使用的 patch 数量（默认 1）
        use_learnable_srm: 是否使用可学习的 SRM 层替代经典 SRM
        learnable_srm_config: 可学习 SRM 的配置字典
    """
    def __init__(
        self,
        pretrain=False,
        topk: int = 1,
        use_learnable_srm: bool = False,
        learnable_srm_config: Optional[dict] = None,
        fusion_mode: str = 'replace',
    ):
        super().__init__()
        self.topk = topk
        self.use_learnable_srm = use_learnable_srm
        self.fusion_mode = fusion_mode
        
        assert fusion_mode in ['replace', 'concat', 'dual_stream'], \
            f"fusion_mode must be 'replace', 'concat', or 'dual_stream', got {fusion_mode}"
        
        # 默认配置
        if learnable_srm_config is None:
            learnable_srm_config = {
                'out_channels': 12,
                'kernel_size': 5,
                'block_type': 'srm_only',
                'freeze_init_epochs': 0,
            }
        self.learnable_srm_config = learnable_srm_config
        
        # 高频滤波层
        if use_learnable_srm:
            print("✓ Using LearnableSRM instead of classic SRMConv2d")
            self.srm = create_learnable_srm(
                in_channels=3,
                out_channels=learnable_srm_config.get('out_channels', 12),
                kernel_size=learnable_srm_config.get('kernel_size', 5),
                block_type=learnable_srm_config.get('block_type', 'srm_only'),
                freeze_init_epochs=learnable_srm_config.get('freeze_init_epochs', 0),
                use_norm=learnable_srm_config.get('use_norm', False),
                use_mixing=learnable_srm_config.get('use_mixing', False),
                seed=learnable_srm_config.get('seed', 42),
            )
            # LearnableSRM 的输出通道数
            self.srm_out_channels = learnable_srm_config.get('out_channels', 12)
        else:
            print("✓ Using classic SRMConv2d")
            self.srm = SRMConv2d_simple()
            # SRMConv2d_simple 输出 64 通道
            self.srm_out_channels = 64
        
        # Fusion layer：如果使用 concat 模式，需要一个小卷积层来融合 SRM + RGB
        if fusion_mode == 'concat':
            fusion_in_channels = self.srm_out_channels + 3  # SRM + RGB
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_in_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            print(f"✓ Using concat fusion mode: SRM({self.srm_out_channels}) + RGB(3) -> fusion_conv -> ResNet")
            resnet_in_channels = 64
        elif fusion_mode == 'dual_stream':
            # 双流模式：SRM 和 RGB 分别通过 conv 后 concat
            # 添加 LayerNorm 确保两个分支的输出尺度一致
            self.srm_stream = nn.Sequential(
                nn.Conv2d(self.srm_out_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, 32),  # 使用 GroupNorm 而非 BN，更稳定
                nn.ReLU(inplace=True),
            )
            self.rgb_stream = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, 32),
                nn.ReLU(inplace=True),
            )
            print(f"✓ Using dual_stream fusion mode: SRM stream + RGB stream -> concat(64) with GroupNorm")
            resnet_in_channels = 64
        else:  # replace
            print(f"✓ Using replace fusion mode: SRM only -> ResNet")
            resnet_in_channels = self.srm_out_channels
        
        # Backbone
        self.disc = resnet50(pretrained=pretrain)
        
        # 调整 ResNet50 的输入通道
        if resnet_in_channels != 3:
            self.disc.conv1 = nn.Conv2d(
                resnet_in_channels,
                64,  # 输出通道：保持 ResNet 设计
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            # 使用 Kaiming 初始化
            nn.init.kaiming_normal_(self.disc.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(f"✓ ResNet conv1 initialized with Kaiming (in_channels={resnet_in_channels})")
        
        # 输出层
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: (B, C, H, W) - 单个 patch
               或 (B*K, C, H, W) - 多个 patch 堆叠
        
        输出:
            logits: (B*K, 1) 或 (B, 1)
        """
        # 只在必要时才进行插值，避免破坏高频噪声结构
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x_resized = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # 高频滤波
        srm_features = self.srm(x_resized)
        
        # 融合模式
        if self.fusion_mode == 'replace':
            # 仅使用 SRM 特征
            features = srm_features
        elif self.fusion_mode == 'concat':
            # SRM + RGB concat
            features = torch.cat([srm_features, x_resized], dim=1)
            features = self.fusion_conv(features)
        elif self.fusion_mode == 'dual_stream':
            # 双流处理后 concat
            srm_out = self.srm_stream(srm_features)
            rgb_out = self.rgb_stream(x_resized)
            features = torch.cat([srm_out, rgb_out], dim=1)
        
        # 分类
        logits = self.disc(features).view(-1, 1)  # (B*K, 1) 或 (B, 1)
        return logits
    
    def freeze_srm(self):
        """冻结 SRM 层的参数"""
        if hasattr(self.srm, 'freeze'):
            self.srm.freeze()
        else:
            for param in self.srm.parameters():
                param.requires_grad = False
    
    def unfreeze_srm(self):
        """解冻 SRM 层的参数"""
        if hasattr(self.srm, 'unfreeze'):
            self.srm.unfreeze()
        else:
            for param in self.srm.parameters():
                param.requires_grad = True
    
    def get_param_groups(self, base_lr: float, srm_lr_scale: float = 1.0, weight_decay: float = 1e-4):
        """
        获取分离的参数组用于 optimizer 初始化
        
        参数：
            base_lr: 基础学习率
            srm_lr_scale: SRM 层学习率缩放因子（默认 1.0）
            weight_decay: 权重衰减
        
        返回：
            参数组列表，用于 optimizer 初始化
        
        使用示例：
            param_groups = model.get_param_groups(base_lr=1e-4, srm_lr_scale=0.1)
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        """
        if not self.use_learnable_srm or srm_lr_scale == 1.0:
            # 如果不使用 learnable SRM 或学习率相同，返回单个参数组
            return [{'params': self.parameters(), 'lr': base_lr}]
        
        # 分离 SRM 和其他参数
        srm_params = list(self.srm.parameters())
        other_params = [p for p in self.parameters() if p not in set(srm_params)]
        
        param_groups = [
            {'params': srm_params, 'lr': base_lr * srm_lr_scale, 'name': 'srm'},
            {'params': other_params, 'lr': base_lr, 'name': 'other'},
        ]
        
        print(f"✓ Created param groups: SRM lr={base_lr * srm_lr_scale:.6e}, Others lr={base_lr:.6e}")
        return param_groups


if __name__ == '__main__':
    model = ssp(pretrain=False)
    print(model)
