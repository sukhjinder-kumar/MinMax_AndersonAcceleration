'''
Torch model definition
'''

import torch
import torch.nn as nn

class TestNN(nn.Module):
    '''
    For MNIST dataset
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
