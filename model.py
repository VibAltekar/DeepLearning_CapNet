import numpy as np
from utils import ReluMarginloss, one_hot, squash
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class CapNet(nn.Module):
    def __init__(self,batch_size,routing_iter,epsilon):
        super(CapNet, self).__init__()
        self.batch_size = batch_size
        self.routing_iter = routing_iter
        self.epsilon = epsilon
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.caps_1 = nn.Conv2d(256, 32*8, kernel_size=9, stride=2)
        w = torch.Tensor(1, 1152, 10, 8, 16)
        nn.init.normal(w)
        self.W = nn.Parameter(w)
        b = torch.zeros(1, 1, 10, 16, 1)
        self.bias = nn.Parameter(b)
        self.recon_fc_1 = nn.Linear(16, 512)
        self.recon_fc_2 = nn.Linear(512, 1024)
        self.recon_fc_3 = nn.Linear(1024, 784)
        self.fc_debug_0 = nn.Linear(160, 50)
        self.fc_debug_1 = nn.Linear(50, 10)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.caps_1(x))
        b_ij = Variable(torch.Tensor(self.batch_size, 1152, 10, 1, 1), requires_grad=False)
        x = self.routing(x, b_ij, self.W,batch_size=self.batch_size,routing_iter=self.routing_iter)
        x = torch.squeeze(x, dim=1)
        v_length = torch.sqrt(torch.sum(torch.mul(x, x),
                                        dim=2, keepdim=True) + self.epsilon)
        v_length = v_length.view(self.batch_size, 10, 1, 1)

        masked_v = torch.matmul(torch.squeeze(x).view(self.batch_size, 16, 10), y.view(-1, 10, 1))

        vector_j = masked_v.view(self.batch_size, 16)
        fc1 = self.recon_fc_1(vector_j)
        fc2 = self.recon_fc_2(fc1)
        reconstruction = F.sigmoid(self.recon_fc_3(fc2))

        return v_length, reconstruction

        x = x.view(-1, 160)
        x = F.relu(self.fc_debug_0(x))

        x = self.fc_debug_1(x)
        return F.log_softmax(x)

    def routing(self, x, b_IJ, W,batch_size,routing_iter):
        x1 = x.view(batch_size, 256, 1, 6, 6)
        x_tile = x1.repeat(1, 1, 10, 1, 1)
        x_view = x_tile.view(batch_size, 1152, 10, 8, 1)
        stride_i = W.repeat(batch_size, 1, 1, 1, 1)
        stride_j = stride_i.view(batch_size, 1152, 10, 16, 8)
        dot_op = torch.matmul(stride_j, x_view)
        dot_op_stopped = Variable(dot_op.data.clone(), requires_grad=False)

        for r_iter in range(routing_iter):
            id_capsule = F.softmax(b_IJ, dim=2)
            if r_iter == routing_iter - 1:
                route_I = torch.mul(id_capsule, dot_op)
                route_I_sum = torch.sum(route_I, dim=1, keepdim=True) + self.bias
                V_J = squash(route_I_sum,self.epsilon)
            if r_iter < routing_iter - 1:

                dot_op_stopped_tmp = dot_op_stopped.data.numpy()
                dot_op_stopped_tmp = np.reshape(dot_op_stopped_tmp, (batch_size, 1152, 10, 16, 1))
                id_capsule_tmp = id_capsule.data.numpy()
                route_I_tmp = id_capsule_tmp * dot_op_stopped_tmp
                route_I_tmp_sum = np.sum(route_I_tmp, axis=1, keepdims=True) + self.bias.data.numpy()
                V_J_tmp = squash(torch.Tensor(route_I_tmp_sum),self.epsilon)

                V_J_tmp_tiled = np.tile(V_J_tmp.numpy(), (1, 1152, 1, 1, 1))
                dot_op_stopped_tmp = np.reshape(dot_op_stopped_tmp, (batch_size, 1152, 10, 1, 16))

                u_produce_v = np.matmul(dot_op_stopped_tmp, V_J_tmp_tiled)

                b_IJ.data += torch.Tensor(u_produce_v)

        return V_J
