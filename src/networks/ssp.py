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
        pretrain=True,
        topk: int = 1,
        use_learnable_srm: bool = False,
        learnable_srm_config: Optional[dict] = None,
    ):
        super().__init__()
        self.topk = topk
        self.use_learnable_srm = use_learnable_srm
        
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
            )
            # LearnableSRM 的输出通道数
            self.srm_out_channels = learnable_srm_config.get('out_channels', 12)
        else:
            print("✓ Using classic SRMConv2d")
            self.srm = SRMConv2d_simple()
            # SRMConv2d_simple 输出 64 通道
            self.srm_out_channels = 64
        
        # Backbone
        self.disc = resnet50(pretrained=pretrain)
        
        # 如果使用 LearnableSRM，需要调整 ResNet50 的输入通道
        if use_learnable_srm:
            # LearnableSRM 输出通道数，需要改 ResNet50 的首层
            original_conv = self.disc.conv1
            self.disc.conv1 = nn.Conv2d(
                self.srm_out_channels,  # 输入通道：SRM 输出
                64,  # 输出通道：保持 ResNet 设计
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            # 复制权重（处理通道数差异）
            if original_conv.weight.shape[1] == 3:
                # 如果原来是 RGB，平均权重到新的输入通道数
                new_weight = original_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                new_weight = new_weight.repeat(1, self.srm_out_channels, 1, 1)  # (64, srm_out, 7, 7)
                self.disc.conv1.weight.data.copy_(new_weight)
        
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
        # 统一尺寸到 256x256
        x = F.interpolate(x, (256, 256), mode='bilinear')
        
        # 高频滤波
        x = self.srm(x)
        
        # 分类
        logits = self.disc(x).view(-1, 1)  # (B*K, 1) 或 (B, 1)
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
    
    def set_srm_learning_rate(self, optimizer, scale: float):
        """
        为 SRM 层设置特定的学习率缩放因子
        
        使用方式：
            model.set_srm_learning_rate(optimizer, 0.1)  # SRM 学习率为基础的 10%
        """
        if not self.use_learnable_srm:
            return
        
        # 获取 SRM 的参数
        srm_params = set(self.srm.parameters())
        
        # 修改参数组的学习率
        for param_group in optimizer.param_groups:
            params = set(param_group['params'])
            
            # 如果这个参数组包含 SRM 参数
            if params & srm_params:
                param_group['lr'] = optimizer.defaults['lr'] * scale
                print(f"✓ Set SRM learning rate to {param_group['lr']:.6f} (scale={scale})")


if __name__ == '__main__':
    model = ssp(pretrain=False)
    print(model)
