import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 10 Dim Normal Data 


class CTDatasetNormal(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x/255
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


normal10DimTrainDl = CTDatasetNormal()
normal10DimTestDl = CTDatasetNormal()
