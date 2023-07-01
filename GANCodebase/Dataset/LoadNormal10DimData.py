'''
Use case:
>>> from Dataset.LoadNormal10DimData import normal10DimTrainDs,normal10DimTestDs,normal10DimTrainDl,normal10DimTestDl
>>>
>>> x = normal10DimTrainDs[0]
>>> x.shape
torch.Size([10])
>>> x = normal10DimTrainDs[0:2] # Slicing allowed
>>> x.shape
torch.Size([2, 10])
>>> for batch in normal10DimTrainDl:
...     # As only one element we can just directly call it, no list[0]
...     print(batch.shape)
...     break
...
torch.Size([5, 10])
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 10 Dim Normal Data 


class DatasetNormal(Dataset):
    # Params
    mean = 0
    stdDev = 1
    numSample = 60000
    dim = 10
    def __init__(self):
        super(Dataset, self).__init__()
        self.x = torch.normal(self.mean,self.stdDev,size=(self.numSample,self.dim))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx]


normal10DimTrainDs = DatasetNormal()
normal10DimTestDs = DatasetNormal()

normal10DimTrainDl = DataLoader(normal10DimTrainDs, batch_size=5, shuffle=True)
normal10DimTestDl = DataLoader(normal10DimTestDs, batch_size=5, shuffle=True)
