import torch.nn as nn


def build_loss():
    return nn.BCEWithLogitsLoss()
