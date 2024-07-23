import torch.nn as nn 
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1.0
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        inter = (y_pred * y_true).sum()
        dsc = (2. * inter + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc