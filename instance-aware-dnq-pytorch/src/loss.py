import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropySmooth(nn.Module):
    """
    Cross entropy loss with label smoothing.
    """
    def __init__(self, smooth_factor, num_classes):
        super(CrossEntropySmooth, self).__init__()
        self.smooth_factor = smooth_factor
        self.num_classes = num_classes

    def forward(self, input, target):
        # Convert target to one-hot encoding
        one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)

        # Apply label smoothing
        smooth_target = (1 - self.smooth_factor) * one_hot + self.smooth_factor / self.num_classes

        # Compute log softmax
        log_probs = F.log_softmax(input, dim=1)

        # Compute cross entropy
        loss = -(smooth_target * log_probs).sum(dim=1).mean()

        return loss
