import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import ReluMarginloss, one_hot
from model import CapNet

epsilon = 1e-9
batch_size = 100
routing_iter = 3
m_plus_value = 0.9
m_minus_value = 0.1
lambda_value = 0.5



def load_data(data_loc="~/.data"):
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_loc, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST(data_loc, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                            batch_size=batch_size, shuffle=True)
    return train_loader,test_loader
def train(train_data,model):
    model.train()
    epoch = 1
    for batch_idx, (data, target) in enumerate(train_data):
        target = torch.Tensor(one_hot(target))
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, recon = model(data, target)
        loss = ReluMarginloss(output, target, data, recon)
        loss.backward()
        optimizer.step()
        print("Epoch: {} batch_idx: {} Loss: {}".format(epoch,batch_idx,loss.data[0]))
        print("Training Complete")
    return model

def test(test_data,model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_data:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data,target)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print("Test Loss: {}".format(test_loss))
    print("Testing Complete")

    test_loss /= len(test_data.dataset)
    print('Loss: {}, Accuracy: {:.0f}%'.format(test_loss, 100. * correct / len(test_data.dataset)))
    return test_loss


if __name__ == "__main__":
    model = CapNet(batch_size=batch_size,routing_iter=routing_iter,epsilon=epsilon)
    optimizer = optim.Adam(model.parameters())
    for param in model.parameters():
        print(param.size())
    train_data,test_data = load_data()
    trained_model = train(train_data,model)
    final_test_loss = test(test_data,trained_model)
