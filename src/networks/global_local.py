"""
Global-Local Dual Stream Network for Forensic Image Classification

架构设计：
1. Local Stream: Patch-based SRM + RGB fusion (现有架构)
   - 捕获局部纹理、伪影、高频细节
   
2. Global Stream: Full-image RGB processing
   - 捕获全局布局、composition、长距离依赖
   
3. Feature Fusion: Concat + FC
   - 结合局部和全局信息
   
性能提升：
- 预期 +2~5% 准确率
- 对 layout/composition 攻击更鲁棒
"""

import torch
from torch import nn
from networks.resnet import resnet50
from networks.srm_conv import SRMConv2d_simple
from networks.learnable_srm import create_learnable_srm
import torch.nn.functional as F
from typing import Optional


class GlobalLocalDualStream(nn.Module):
    """
    Global-Local Dual Stream 网络
    
    参数：
        pretrain: 是否使用预训练权重
        topk: local stream 的 patch 数量
        use_learnable_srm: 是否使用可学习 SRM
        learnable_srm_config: SRM 配置
        fusion_mode: local stream 的融合模式 ('replace', 'concat', 'dual_stream')
        global_size: global stream 的输入尺寸 (推荐 384 或 512)
        share_backbone: local 和 global 是否共享 ResNet 权重
        fusion_type: 特征融合方式 ('concat', 'add', 'attention')
    """
    
    def __init__(
        self,
        pretrain: bool = False,
        topk: int = 1,
        use_learnable_srm: bool = False,
        learnable_srm_config: Optional[dict] = None,
        fusion_mode: str = 'replace',
        global_size: int = 384,
        share_backbone: bool = False,
        fusion_type: str = 'concat',
    ):
        super().__init__()
        self.topk = topk
        self.use_learnable_srm = use_learnable_srm
        self.fusion_mode = fusion_mode
        self.global_size = global_size
        self.share_backbone = share_backbone
        self.fusion_type = fusion_type
        
        print("="*60)
        print("Initializing Global-Local Dual Stream Network")
        print("="*60)
        
        # ============== Local Stream ==============
        print("\n[Local Stream]")
        
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
            print("✓ Using LearnableSRM")
            self.local_srm = create_learnable_srm(
                in_channels=3,
                out_channels=learnable_srm_config.get('out_channels', 12),
                kernel_size=learnable_srm_config.get('kernel_size', 5),
                block_type=learnable_srm_config.get('block_type', 'srm_only'),
                freeze_init_epochs=learnable_srm_config.get('freeze_init_epochs', 0),
                use_norm=learnable_srm_config.get('use_norm', False),
                use_mixing=learnable_srm_config.get('use_mixing', False),
                seed=learnable_srm_config.get('seed', 42),
            )
            self.srm_out_channels = learnable_srm_config.get('out_channels', 12)
        else:
            print("✓ Using classic SRMConv2d")
            self.local_srm = SRMConv2d_simple()
            self.srm_out_channels = 64
        
        # Local fusion layer
        if fusion_mode == 'concat':
            fusion_in_channels = self.srm_out_channels + 3
            self.local_fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_in_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            print(f"✓ Local fusion: concat (SRM + RGB)")
            local_resnet_in = 64
        elif fusion_mode == 'dual_stream':
            self.local_srm_stream = nn.Sequential(
                nn.Conv2d(self.srm_out_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.local_rgb_stream = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            print(f"✓ Local fusion: dual_stream")
            local_resnet_in = 64
        else:  # replace
            print(f"✓ Local fusion: replace (SRM only)")
            local_resnet_in = self.srm_out_channels
        
        # Local backbone
        self.local_backbone = resnet50(pretrained=pretrain)
        
        # 调整 local backbone 输入通道
        if local_resnet_in != 3:
            self.local_backbone.conv1 = nn.Conv2d(
                local_resnet_in, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(self.local_backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(f"✓ Local ResNet conv1: {local_resnet_in} → 64")
        
        # 移除 local backbone 的 FC 层
        self.local_feature_dim = self.local_backbone.fc.in_features
        self.local_backbone.fc = nn.Identity()
        
        # ============== Global Stream ==============
        print("\n[Global Stream]")
        
        if share_backbone:
            print("✓ Sharing backbone with local stream")
            self.global_backbone = self.local_backbone
            # 如果共享 backbone 且 local 修改了 conv1，需要为 global 添加适配层
            if local_resnet_in != 3:
                print(f"✓ Adding RGB adapter for global stream (3 → {local_resnet_in})")
                self.global_adapter = nn.Sequential(
                    nn.Conv2d(3, local_resnet_in, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(local_resnet_in),
                    nn.ReLU(inplace=True),
                )
            else:
                self.global_adapter = None
        else:
            print("✓ Using separate backbone")
            self.global_backbone = resnet50(pretrained=pretrain)
            self.global_feature_dim = self.global_backbone.fc.in_features
            self.global_backbone.fc = nn.Identity()
            self.global_adapter = None
        
        self.global_feature_dim = 2048  # ResNet50 输出维度
        
        print(f"✓ Global input size: {global_size}x{global_size}")
        
        # ============== Feature Fusion ==============
        print("\n[Feature Fusion]")
        
        if fusion_type == 'concat':
            # 简单拼接
            fusion_dim = self.local_feature_dim + self.global_feature_dim
            self.fusion_fc = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
            )
            print(f"✓ Fusion type: concat ({self.local_feature_dim} + {self.global_feature_dim} → 512 → 1)")
        
        elif fusion_type == 'add':
            # 加权相加
            assert self.local_feature_dim == self.global_feature_dim, \
                "For 'add' fusion, local and global features must have same dim"
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.fusion_fc = nn.Linear(self.local_feature_dim, 1)
            print(f"✓ Fusion type: add (learnable weights)")
        
        elif fusion_type == 'attention':
            # 注意力融合
            self.local_attention = nn.Sequential(
                nn.Linear(self.local_feature_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
            self.global_attention = nn.Sequential(
                nn.Linear(self.global_feature_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
            fusion_dim = self.local_feature_dim + self.global_feature_dim
            self.fusion_fc = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
            )
            print(f"✓ Fusion type: attention (with learnable attention weights)")
        
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        print("\n" + "="*60)
        print(f"✓ Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("="*60 + "\n")
    
    def forward(self, x_local, x_global):
        """
        前向传播
        
        输入:
            x_local: (B*K, 3, 256, 256) - Local patches
            x_global: (B, 3, global_size, global_size) - Global full images
        
        输出:
            logits: (B*K, 1)
        
        注意：
        - x_local 已经是展平后的 (B*K, C, H, W)
        - x_global 是原始 batch 的全局图 (B, C, H, W)
        - 需要将 global_features 扩展到 (B*K) 来匹配 local_features
        """
        # 计算 batch_size 和 num_patches
        batch_size = x_global.size(0)
        num_local = x_local.size(0)
        num_patches = num_local // batch_size
        
        # ============== Local Stream ==============
        # 确保尺寸为 256x256（通常已经是，但保险起见）
        if x_local.shape[-1] != 256 or x_local.shape[-2] != 256:
            x_local = F.interpolate(x_local, (256, 256), mode='bilinear', align_corners=False)
        
        # 高频滤波
        srm_features = self.local_srm(x_local)
        
        # 融合模式
        if self.fusion_mode == 'replace':
            local_input = srm_features
        elif self.fusion_mode == 'concat':
            local_input = torch.cat([srm_features, x_local], dim=1)
            local_input = self.local_fusion_conv(local_input)
        elif self.fusion_mode == 'dual_stream':
            srm_out = self.local_srm_stream(srm_features)
            rgb_out = self.local_rgb_stream(x_local)
            local_input = torch.cat([srm_out, rgb_out], dim=1)
        
        # Local feature extraction
        local_features = self.local_backbone(local_input)  # (B*K, 2048)
        
        # ============== Global Stream ==============
        # 确保尺寸为 global_size（通常已经是，但保险起见）
        if x_global.shape[-1] != self.global_size or x_global.shape[-2] != self.global_size:
            x_global = F.interpolate(x_global, (self.global_size, self.global_size), 
                                     mode='bilinear', align_corners=False)
        
        # 如果共享 backbone 且需要通道适配
        if self.global_adapter is not None:
            x_global = self.global_adapter(x_global)
        
        # Global feature extraction
        global_features = self.global_backbone(x_global)  # (B, 2048)
        
        # ============== 维度对齐 ==============
        # 将 global_features 从 (B, Dim) 扩展到 (B*K, Dim)
        # 方法: (B, Dim) -> (B, 1, Dim) -> (B, K, Dim) -> (B*K, Dim)
        global_features_expanded = global_features.unsqueeze(1).repeat(1, num_patches, 1).view(num_local, -1)  # (B*K, 2048)
        
        # ============== Feature Fusion ==============
        if self.fusion_type == 'concat':
            # 简单拼接
            fused_features = torch.cat([local_features, global_features_expanded], dim=1)  # (B*K, 4096)
            logits = self.fusion_fc(fused_features)
        
        elif self.fusion_type == 'add':
            # 加权相加
            weights = F.softmax(self.fusion_weight, dim=0)
            fused_features = weights[0] * local_features + weights[1] * global_features_expanded
            logits = self.fusion_fc(fused_features)
        
        elif self.fusion_type == 'attention':
            # 注意力融合
            local_att = self.local_attention(local_features)  # (B*K, 1)
            global_att = self.global_attention(global_features_expanded)  # (B*K, 1)
            
            att_weights = F.softmax(torch.cat([local_att, global_att], dim=1), dim=1)  # (B*K, 2)
            
            # 加权特征
            weighted_local = local_features * att_weights[:, 0:1]
            weighted_global = global_features_expanded * att_weights[:, 1:2]
            
            fused_features = torch.cat([weighted_local, weighted_global], dim=1)
            logits = self.fusion_fc(fused_features)
        
        return logits.view(-1, 1)
    
    def freeze_srm(self):
        """冻结 SRM 层的参数"""
        if hasattr(self.local_srm, 'freeze'):
            self.local_srm.freeze()
        else:
            for param in self.local_srm.parameters():
                param.requires_grad = False
    
    def unfreeze_srm(self):
        """解冻 SRM 层的参数"""
        if hasattr(self.local_srm, 'unfreeze'):
            self.local_srm.unfreeze()
        else:
            for param in self.local_srm.parameters():
                param.requires_grad = True
    
    def get_param_groups(self, base_lr: float, srm_lr_scale: float = 1.0, weight_decay: float = 1e-4):
        """
        获取分离的参数组用于 optimizer 初始化
        """
        if not self.use_learnable_srm or srm_lr_scale == 1.0:
            return [{'params': self.parameters(), 'lr': base_lr}]
        
        # 分离 SRM 和其他参数
        srm_params = list(self.local_srm.parameters())
        other_params = [p for p in self.parameters() if p not in set(srm_params)]
        
        param_groups = [
            {'params': srm_params, 'lr': base_lr * srm_lr_scale, 'name': 'srm'},
            {'params': other_params, 'lr': base_lr, 'name': 'other'},
        ]
        
        print(f"✓ Created param groups: SRM lr={base_lr * srm_lr_scale:.6e}, Others lr={base_lr:.6e}")
        return param_groups


# 便捷工厂函数
def create_global_local_model(
    model_type: str = 'standard',
    pretrain: bool = False,
    use_learnable_srm: bool = True,
    **kwargs
) -> GlobalLocalDualStream:
    """
    创建 Global-Local 模型的工厂函数
    
    model_type:
        - 'standard': 标准配置（推荐）
        - 'lightweight': 轻量级（共享 backbone）
        - 'high_performance': 高性能（attention fusion）
    """
    if model_type == 'standard':
        return GlobalLocalDualStream(
            pretrain=pretrain,
            topk=1,
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config={
                'out_channels': 12,
                'kernel_size': 5,
                'block_type': 'srm_only',
                'use_norm': True,
                'use_mixing': False,
            },
            fusion_mode='concat',
            global_size=384,
            share_backbone=False,
            fusion_type='concat',
            **kwargs
        )
    
    elif model_type == 'lightweight':
        return GlobalLocalDualStream(
            pretrain=pretrain,
            topk=1,
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config={
                'out_channels': 8,
                'kernel_size': 3,
                'block_type': 'srm_only',
                'use_norm': False,
            },
            fusion_mode='replace',
            global_size=320,
            share_backbone=True,  # 共享权重
            fusion_type='add',
            **kwargs
        )
    
    elif model_type == 'high_performance':
        return GlobalLocalDualStream(
            pretrain=pretrain,
            topk=3,  # 使用 3 个 patch
            use_learnable_srm=use_learnable_srm,
            learnable_srm_config={
                'out_channels': 16,
                'kernel_size': 5,
                'block_type': 'srm_only',
                'use_norm': True,
                'use_mixing': True,
            },
            fusion_mode='dual_stream',
            global_size=512,
            share_backbone=False,
            fusion_type='attention',  # 注意力融合
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    model = create_global_local_model('standard', pretrain=False)
    B, K = 4, 3
    x_local = torch.randn(B * K, 3, 256, 256)
    x_global = torch.randn(B, 3, 384, 384)
    out = model(x_local, x_global)
    print(out.shape)  # (B*K, 1)

