import numpy as np
import torch
import torch.nn as nn


class DotDict(dict):
    """A dictionary that supports dot notation access."""
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, iter, total_num_iter, initial_lr):
    if not isinstance(initial_lr, list):
        initial_lr = [initial_lr] * len(optimizer.param_groups)

    for param_group, init_lr in zip(optimizer.param_groups, initial_lr):
        lr = init_lr - (init_lr * (iter / float(total_num_iter)))
        param_group['lr'] = lr


def update_linear_schedule_with_warmup(optimizer, iter, total_num_iter, initial_lr, warmup_iter):
    if not isinstance(initial_lr, list):
        initial_lr = [initial_lr] * len(optimizer.param_groups)
    
    for param_group, init_lr in zip(optimizer.param_groups, initial_lr):
        if iter < warmup_iter:
            # Linearly increase LR during warmup
            lr = init_lr * (iter / float(warmup_iter))
        else:
            # Linearly decay LR after warmup
            decay_epochs = total_num_iter - warmup_iter
            lr = init_lr - (init_lr * ((iter - warmup_iter) / float(decay_epochs)))
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def compute_velocity(infos, device):
    velocities = torch.zeros(len(infos), 1).to(device)
    for idx, info in enumerate(infos):
        if 'y_velocity' not in infos:
            velocity = np.abs(info['x_velocity'])
        else:
            velocity = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
        velocities[idx] = velocity
    return velocities
