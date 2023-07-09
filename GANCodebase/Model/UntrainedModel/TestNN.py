'''
Torch model definition

Use case:
>>> from Model.UntrainedModel.TestNN import TestNN
>>>
>>> f = TestNN()
>>> x = mnistTrainDs[0][0]
>>> x.shape
torch.Size([1, 28, 28])
>>> f(x)
tensor([-0.0520, -0.0050, -0.1206, -0.0306, -0.1516,  0.0956,  0.0816, -0.0283,
        -0.0700,  0.0969], grad_fn=<SqueezeBackward0>)
>>> # Ideally pass x.view(1,28*28)
>>> f(x.view(1,28*28))
tensor([-0.0520, -0.0050, -0.1206, -0.0306, -0.1516,  0.0956,  0.0816, -0.0283,
        -0.0700,  0.0969], grad_fn=<SqueezeBackward0>)
'''

import torch
import torch.nn as nn


class TestNN(nn.Module):
    '''
    For MNIST dataset. Take (-1,28*28) tensors
    '''
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28,100)
        self.layer2 = nn.Linear(100,50)
        self.layer3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.R(self.layer1(x))
        x = self.R(self.layer2(x))
        x = self.layer3(x)
        return x.squeeze()
