import torch
import torch.nn as nn

from utils.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = AddBias(torch.zeros(num_outputs))

    def forward(self, x, device):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())

        if x.is_cuda:
            zeros = zeros.to(device)

        action_logstd = self.log_std(zeros)

        return FixedNormal(action_mean, action_logstd.exp() * 0.5)
