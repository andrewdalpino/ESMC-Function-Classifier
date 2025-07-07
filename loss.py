import torch

from torch.nn import Module


class ZLPRLoss(Module):
    """Zero-bounded log-sum-exp and pairwise ranking loss for multi-label classification."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        positive = torch.log(1 + torch.inner(y, torch.exp(-y_pred)))
        negative = torch.log(1 + torch.inner(1 - y, torch.exp(y_pred)))

        loss = torch.mean(positive + negative)

        return loss
