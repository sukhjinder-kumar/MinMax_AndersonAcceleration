'''
For each dataset have train and test Dataset and Dataloader
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F

# MNIST

mnistPath = "./Dataset/DatasetFiles"
transform = transforms.Compose([
    transforms.ToTensor(), # Convert to tensor
    transforms.Normalize((0,), (1,)) # Mean 0 and Std Dev 1
])
targetTransform = transforms.Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one hot encoding

mnistTrainDs =  MNIST(mnistPath, train=True, download=True, 
                      transform=transform, target_transform=targetTransform)
mnistTestDs = MNIST(mnistPath, train=False, download=True, 
                    transform=transform, target_transform=targetTransform)
mnistTrainDl = DataLoader(mnistTrainDs, batch_size=5, shuffle=True)
mnistTestDl = DataLoader(mnistTestDs, batch_size=5, shuffle=True)
