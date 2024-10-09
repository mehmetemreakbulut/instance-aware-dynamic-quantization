import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbleSoftmax(nn.Module):
    def __init__(self, hard=True):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = torch.rand_like(template_tensor)
        gumbel_samples_tensor = -torch.log(-torch.log(uniform_samples_tensor + eps) + eps)
        return gumbel_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        gumbel_samples_tensor = self.sample_gumbel_like(logits)
        gumbel_trick_log_prob_samples = (logits + gumbel_samples_tensor) / temperature
        soft_samples = F.softmax(gumbel_trick_log_prob_samples, dim=-1)
        return soft_samples

    def gumbel_softmax(self, logits, temperature=1.0, hard=True):
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            index = y.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y.detach() + y
        else:
            ret = y
        return ret

    def forward(self, logits, temp=1, force_hard=True):
        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=temp, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=temp, hard=True)
