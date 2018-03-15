import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def one_hot(labels):
    out = []
    for i, label in enumerate(labels):
        out.append(np.zeros(10))
        out[i][label] = 1
    return np.array(out)
def squash(vector,eps):
    vec_squared_norm = torch.sum((vector * vector), dim=-2, keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + eps)
    return (scalar_factor * vector)

def ReluMarginloss(m_plus_value,m_minus_value,v_length, target, x, recon):
    left = F.relu(m_plus_value - v_length, inplace=True) ** 2
    right = F.relu(v_length - m_minus_value, inplace=True) ** 2
    margin_loss = target * left + 0.5 * (1. - target) * right
    margin_loss = torch.mean(torch.sum(margin_loss, dim=1))
    recon_loss = nn.MSELoss(size_average=False)
    loss = (margin_loss + 0.005 * recon_loss(recon, x)) / x.size(0)
    return loss
