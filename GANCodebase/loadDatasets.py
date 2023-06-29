'''
For each dataset have train and test Dataset and Dataloader
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# MNIST

mnistPath = "./DatasetFiles/MNIST"
mnistTrainPath = mnistPath + "/training.pt"
mnistTestPath = mnistPath + "/test.pt"

class CTDatasetMnist(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x/255
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

mnistTrainDs = CTDatasetMnist(mnistTrainPath)
mnistTestDs = CTDatasetMnist(mnistTestPath)

mnistTrainDl = DataLoader(mnistTrainDs, batch_size=5)
mnistTestDl = DataLoader(mnistTestDs, batch_size=5)
