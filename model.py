import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class CapNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv1: [batch_size, 20, 20, 256]
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)

        # Cpasule layer 1: [batch_size, 1152, 8, 1]
        self.caps_1 = nn.Conv2d(256, 32*8, kernel_size=9, stride=2)

        # Capsule layer 2: [batch_size, 10, 16, 1]
        w = torch.Tensor(1, 1152, 10, 8, 16)
        nn.init.normal(w)
        self.W = nn.Parameter(w)
        b = torch.zeros(1, 1, 10, 16, 1)
        self.bias = nn.Parameter(b)

        # Reconstruction Layers
        self.recon_fc_1 = nn.Linear(16, 512)
        self.recon_fc_2 = nn.Linear(512, 1024)
        self.recon_fc_3 = nn.Linear(1024, 784)

        # Debug flat and full-connected
        self.fc_debug_0 = nn.Linear(160, 50)
        self.fc_debug_1 = nn.Linear(50, 10)

    def forward(self, x, y):
        # conv layer1
        x = F.relu(self.conv1(x))

        # capsule layer1
        x = F.relu(self.caps_1(x))

        # capsule layer2
        b_ij = Variable(torch.Tensor(batch_size, 1152, 10, 1, 1), requires_grad=False)
        x = self.routing(x, b_ij, self.W)
        x = torch.squeeze(x, dim=1)

        # decoder layer
        v_length = torch.sqrt(torch.sum(torch.mul(x, x),
                                        dim=2, keepdim=True) + epsilon)
        v_length = v_length.view(batch_size, 10, 1, 1)

        masked_v = torch.matmul(torch.squeeze(x).view(batch_size, 16, 10), y.view(-1, 10, 1))

        # reconstruction layer
        vector_j = masked_v.view(batch_size, 16)
        fc1 = self.recon_fc_1(vector_j)
        fc2 = self.recon_fc_2(fc1)
        reconstruction = F.sigmoid(self.recon_fc_3(fc2))

        return v_length, reconstruction

        x = x.view(-1, 160)
        x = F.relu(self.fc_debug_0(x))

        x = self.fc_debug_1(x)
        return F.log_softmax(x)

    def routing(self, x, b_IJ, W):
        # Tiling input
        x1 = x.view(batch_size, 256, 1, 6, 6)
        x_tile = x1.repeat(1, 1, 10, 1, 1)
        x_view = x_tile.view(batch_size, 1152, 10, 8, 1)
        W_tile = W.repeat(batch_size, 1, 1, 1, 1)
        W_view = W_tile.view(batch_size, 1152, 10, 16, 8)

        u_hat = torch.matmul(W_view, x_view)

        # clone u_hat for intermediate routing iters
        u_hat_stopped = Variable(u_hat.data.clone(), requires_grad=False)

        # routing
        #print "Start routing..."
        for r_iter in range(routing_iter):
            c_IJ = F.softmax(b_IJ, dim=2)

            # last iteration
            if r_iter == routing_iter - 1:
                s_J = torch.mul(c_IJ, u_hat)
                s_J_sum = torch.sum(s_J, dim=1, keepdim=True) + self.bias
                V_J = squash(s_J_sum)

            # routing ieration
            if r_iter < routing_iter - 1:
                #u_hat_stopped_0 = u_hat_stopped.view(batch_size, 1152, 10, 16, 1)
                #s_J_tmp = torch.mul(c_IJ, u_hat_stopped_0)
                #s_J_tmp_sum = torch.sum(s_J_tmp, dim=1, keepdim=True) + self.bias
                #V_J_tmp = squash(s_J_tmp_sum)

                # Tile V_J
                #V_J_tmp_tiled = V_J_tmp.repeat(1, 1152, 1, 1, 1)
                #u_hat_stopped_1 = u_hat_stopped_0.view(batch_size, 1152, 10, 1, 16)

                # update b_IJ
                #u_produce_v = torch.matmul(u_hat_stopped_1, V_J_tmp_tiled)
            #    assert u_produce_v.size() == (batch_size, 1152, 10, 1, 1)

                #b_IJ += u_produce_v

                # implement with numpy operations
                u_hat_stopped_tmp = u_hat_stopped.data.numpy()
                u_hat_stopped_tmp = np.reshape(u_hat_stopped_tmp, (batch_size, 1152, 10, 16, 1))
                c_IJ_tmp = c_IJ.data.numpy()
                s_J_tmp = c_IJ_tmp * u_hat_stopped_tmp
                s_J_tmp_sum = np.sum(s_J_tmp, axis=1, keepdims=True) + self.bias.data.numpy()
                V_J_tmp = squash(torch.Tensor(s_J_tmp_sum))

                V_J_tmp_tiled = np.tile(V_J_tmp.numpy(), (1, 1152, 1, 1, 1))
                u_hat_stopped_tmp = np.reshape(u_hat_stopped_tmp, (batch_size, 1152, 10, 1, 16))

                u_produce_v = np.matmul(u_hat_stopped_tmp, V_J_tmp_tiled)

                b_IJ.data += torch.Tensor(u_produce_v)

        #print "Finished routing"
        return V_J
