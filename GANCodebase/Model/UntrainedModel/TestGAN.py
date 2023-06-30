import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,50) # dim = 10
        self.layer2 = nn.Linear(50,100)
        self.layer3 = nn.Linear(100, 28*28)
        self.R = nn.ReLU()
    def forward(self,z):
        z = self.R(self.layer1(z))
        z = self.R(self.layer2(z))
        z = self.layer3(z)
        return z.squeeze()

class Discriminator(nn.Module):
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
