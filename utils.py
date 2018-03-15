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
def NLLloss(v_length, target):
    v_length = v_length.view(batch_size, 10)
    logits = F.log_softmax(v_length, dim=1)
    loss = F.nll_loss(logits, target)
    return loss
def MultiMarginloss(v_length, target):
    v_length.data = v_length.data.view(batch_size, 10)
    loss_pos = F.multi_margin_loss(v_length, target, p=2, margin=m_plus_value)
    loss_neg = F.multi_margin_loss(v_length, target, p=2, margin=m_minus_value)
    return loss_neg + loss_pos

def Marginloss(v_length, target):
    print(v_length.size())
    print(target.size())
    m_plus = Variable(torch.Tensor(np.array(np.tile(m_plus_value, [batch_size, 10, 1, 1]))))
    m_minus = Variable(torch.Tensor(np.array(np.tile(m_minus_value, [batch_size, 10, 1, 1]))))
    lambda_val = Variable(torch.Tensor(np.array(np.tile(lambda_value, [batch_size, 10]))))
    zero = Variable(torch.zeros(batch_size, 10, 1, 1))
    ones = Variable(torch.ones(batch_size, 10))

    max_l = m_plus - v_length
    max_l = torch.mul(max_l, max_l)

    max_r = v_length - m_minus
    max_r = torch.mul(max_r, max_r)
    assert max_l.shape == (batch_size, 10, 1, 1)

    max_l.data = max_l.data.view(batch_size, 10)
    max_r.data = max_r.data.view(batch_size, 10)

    T_c = target

    L_c = torch.mul(T_c, max_l) + torch.mul(lambda_val, torch.mul((ones - T_c), max_r))

    margin_loss = torch.mean(torch.sum(L_c, dim=1))

    return margin_loss

def ReluMarginloss(v_length, target, x, recon):
    # margin loss
    left = F.relu(0.9 - v_length, inplace=True) ** 2
    right = F.relu(v_length - 0.1, inplace=True) ** 2

    margin_loss = target * left + 0.5 * (1. - target) * right
    margin_loss = torch.mean(torch.sum(margin_loss, dim=1))

    # reconstrcution loss
    recon_loss = nn.MSELoss(size_average=False)

    loss = (margin_loss + 0.005 * recon_loss(recon, x)) / x.size(0)

    return loss
