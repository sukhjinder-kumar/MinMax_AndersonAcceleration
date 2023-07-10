'''
Use case:

>>> from Dataset.LoadNormalData import DatasetNormal 
>>> from Dataset.LoadGuassianData import guassian8Ds,guassian8Dl,guassian25Ds,guassian25Dl
>>> from Model.UntrainedModel.GuassianGAN import Generator, Discriminator 
>>> batchSize = 128 
>>> dim = 8
>>> G = Generator(dim=dim)
>>> D = Discriminator()
>>> 
>>> guassian8Dl = DataLoader(guassian8Ds, batch_size=batchSize, shuffle=True)
>>> normalDTrainDs = DatasetNormal(dim=dim, numSample=100000)
>>> normalDTrainDl = DataLoader(normalDTrainDs, batch_size=batchSize, shuffle=True)
>>> 
>>> print(normalDTrainDs[0].shape)
torch.Size([8])
>>> print(G(normalDTrainDs[0]))
tensor([ 0.0599, -0.1027], grad_fn=<SqueezeBackward0>)
>>> print(D(guassian8Ds[0]))
tensor(0.5146, grad_fn=<SqueezeBackward0>)
>>> print(D(guassian8Ds[0]).shape)
torch.Size([])

>>> for batch in normalDTrainDl:
...     temp = batch
...     break
...
>>> print(G(temp).shape)
torch.Size([128, 2])

>>> for batch in guassian8Dl:
...     temp = batch
...     break
...
>>> print(temp.shape)
torch.Size([128, 2])
>>> print(D(temp).shape)
torch.Size([128])
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
    

class Generator(nn.Module):
    '''
    Input shape [num,dim=8]
    Output shape [num,2]. If num = 1 than [2,] 
    '''
    def __init__(self, dim=8, hidden_size=128, output_size=2):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(dim, hidden_size)
        self.map2 = nn.Linear(hidden_size, int(hidden_size))
        self.map3 = nn.Linear(hidden_size, int(hidden_size))
        self.map4 = nn.Linear(int(hidden_size), output_size)
        self.f = torch.relu
    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        x = self.f(x)
        x = self.map4(x)
        return x.squeeze()

class Discriminator(nn.Module):
    '''
    Input shape [num,2]
    Output shape [num,1] is reduced to [num]
    '''
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = torch.sigmoid
        self.f2 = torch.tanh
        
    def forward(self, x):
        x = x.view(-1,2)
        x = self.f2(self.map1(x))
        x = self.f2(self.map2(x))
        x = self.f(self.map3(x))
        return x.squeeze()
