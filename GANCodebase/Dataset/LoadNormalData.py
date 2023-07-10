'''
Use case:
>>> from Dataset.LoadNormalData import normalTrainDs,normalTestDs,normalTrainDl,normalTestDl
>>>
>>> x = normalTrainDs[0]
>>> x.shape
torch.Size([10])
>>> x = normalTrainDs[0:2] # Slicing allowed
>>> x.shape
torch.Size([2, 10])
>>> for batch in normalTrainDl:
...     # As only one element we can just directly call it, no list[0]
...     print(batch.shape)
...     break
...
torch.Size([5, 10])
>>> from Dataset.LoadNormalData import DatasetNormal
>>> normalTrainDs = DatasetNormal(dim=100)
>>> normalTestDs = DatasetNormal(dim=100)
>>> normalTrainDl = DataLoader(normalTrainDs, batch_size=5, shuffle=True)
>>> normalTestDl = DataLoader(normalTestDs, batch_size=5, shuffle=True)

'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Default 10 Dim Normal Data 


class DatasetNormal(Dataset):
    # Params
    mean = 0
    stdDev = 1
    def __init__(self, dim=10, numSample=60000):
        super(DatasetNormal, self).__init__()
        self.dim = dim
        self.numSample = numSample
        self.x = torch.normal(self.mean,self.stdDev,size=(self.numSample,self.dim))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx]


normal10DimTrainDs = DatasetNormal()
normal10DimTestDs = DatasetNormal()

normal10DimTrainDl = DataLoader(normal10DimTrainDs, batch_size=5, shuffle=True)
normal10DimTestDl = DataLoader(normal10DimTestDs, batch_size=5, shuffle=True)
