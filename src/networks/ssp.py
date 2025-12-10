import torch
from torch import nn
from networks.resnet import resnet50
from networks.srm_conv import SRMConv2d_simple
import torch.nn.functional as F


class ssp(nn.Module):
    def __init__(self, pretrain=True, topk: int = 1):
        super().__init__()
        self.topk = topk
        self.srm = SRMConv2d_simple()
        self.disc = resnet50(pretrained=True)
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x):
        # x: shape (B, C, H, W) for single patch; (B*K, C, H, W) for stacked top-k patches
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.srm(x)
        logits = self.disc(x).view(-1, 1)  # (B*K, 1)
        return logits


if __name__ == '__main__':
    model = ssp(pretrain=True)
    print(model)
