'''
For each dataset have train and test Dataset and Dataloader

Use case:
>>> from Dataset.LoadMnistDataset import mnistTrainDs,mnistTestDs,mnistTrainDl,mnistTestDl
>>>
>>> x,y = mnistTrainDs[0]
>>> x.shape # Redundant extra axis
torch.Size([1, 28, 28])
>>> y.shape
torch.Size([10])
>>> # slicing not allowed in mnistTrainDs
>>>
>>> for batch in mnistTrainDl:
...     # batch is a list [x,y]
...     print(batch[0].shape)
...     print(batch[1].shape)
...     break
...
torch.Size([5, 1, 28, 28])
torch.Size([5, 10])
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
