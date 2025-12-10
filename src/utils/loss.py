import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLossSmooth(nn.Module):
    """Binary cross entropy with optional label smoothing."""

    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

        smooth_target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(input, smooth_target, reduction=self.reduction)


def build_loss(smoothing: float = 0.0):
    return BCEWithLogitsLossSmooth(smoothing=smoothing)
