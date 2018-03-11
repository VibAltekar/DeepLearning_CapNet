import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import ReluMarginloss
from model import CapNet


epsilon = 1e-9
batch_size = 100
routing_iter = 3
m_plus_value = 0.9
m_minus_value = 0.1
lambda_value = 0.5


model = CapNet()
optimizer = optim.Adam(model.parameters())
for param in model.parameters():
    print(param.size())
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/.data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST('~/.data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                            batch_size=batch_size, shuffle=True)
def train(epoch=1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.Tensor(one_hot(target))
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, recon = model(data, target)
        loss = ReluMarginloss(output, target, data, recon)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    train()
    test()
