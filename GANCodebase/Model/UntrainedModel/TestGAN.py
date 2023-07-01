'''
Use case:
>>> from Model.UntrainedModel.TestGAN import Generator, Discriminator
>>>
>>> G = Generator()
>>> D = Discriminator()
>>> z = normal10DimTrainDs[0:2]
>>> G(z)
tensor([[ 0.0626,  0.0289,  0.0132,  ...,  0.0528,  0.1721, -0.1249],
        [ 0.0335, -0.0845,  0.0130,  ..., -0.0050,  0.0082, -0.1303]],
       grad_fn=<SqueezeBackward0>)
>>> G(z).shape
torch.Size([2, 784])
>>> x = mnistTrainDs[0][0]
>>> D(x.view(-1,28*28))
tensor(0.5477, grad_fn=<SqueezeBackward0>)
>>> D(G(z))
tensor([0.5382, 0.5367], grad_fn=<SqueezeBackward0>)
'''

import torch
import torch.nn as nn


class Generator(nn.Module):
    '''
    Input shape [num,10]
    '''
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,50) # dim = 10
        self.layer2 = nn.Linear(50,100)
        self.layer3 = nn.Linear(100, 28*28)
        self.R = nn.ReLU()
    def forward(self,z):
        z = z.view(-1,10)
        z = self.R(self.layer1(z))
        z = self.R(self.layer2(z))
        z = self.layer3(z)
        return z.squeeze()


class Discriminator(nn.Module):
    '''
    Input shape [num,28*28]
    '''
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28,100)
        self.layer2 = nn.Linear(100,25)
        self.layer3 = nn.Linear(25,1)
        self.R = nn.ReLU()
        self.S = nn.Sigmoid()
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.R(self.layer1(x))
        x = self.R(self.layer2(x))
        x = self.S(self.layer3(x))
        return x.squeeze()
