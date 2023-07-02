import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    '''
    Input shape [num,dim=100]
    Output shape [num,28*28]. If num = 1 than [28*28]
    '''
    def __init__(self, dim=100):
        super(Generator, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024,28*28)

    def forward(self, z):
        z = z.view(-1,self.dim)
        z = F.leaky_relu(self.layer1(z), 0.2)
        z = F.leaky_relu(self.layer2(z), 0.2)
        z = F.leaky_relu(self.layer3(z), 0.2)
        z = F.tanh(self.layer4(z))
        return z.squeeze()


class Discriminator(nn.Module):
    '''
    Input shape [num,28*28]
    Output shape [num,1] is reduced to [num]
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(28*28, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.leaky_relu(self.layer1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.layer2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.layer3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.layer4(x))
        return x.squeeze()
