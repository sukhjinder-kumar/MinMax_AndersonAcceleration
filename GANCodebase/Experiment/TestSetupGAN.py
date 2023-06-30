import sys
sys.path.insert(1,'/Users/sukhkuma/Dev/Programming/Research/MinMax_AA/MinMax_AndersonAcceleration/GANCodebase')

from Model.UntrainedModel.TestGAN import Generator, Discriminator
from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs
from Dataset.LoadNormal10DimData import normal10DimTrainDl, normal10DimTestDl

import torch
import torch.nn as nn
from torch.optim import ADAM 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def TrainModel(dl,G,D,numEpochs):
    pass


# Init model def
G = Generator()
D = Discriminator()

# Take Ds and convert to Dl with same batchSize across both random and real data
batchSize = 5
mnistTrainDl = DataLoader(mnistTrainDs, batch_size=batchSize)
mnistTestDl = DataLoader(mnistTestDs, batch_size=batchSize)
normal10DimTrainDl = DataLoader(normal10DimTrainDl, batch_size=batchSize)
normal10DimTestDl = DataLoader(normal10DimTestDl, batch_size=batchSize)

# Train Model
epochs, gLoss, dLoss = TrainModel(mnistTrainDl,normal10DimTrainDl,G,D,numEpochs=2)
