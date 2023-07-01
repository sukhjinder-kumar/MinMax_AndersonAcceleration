import sys
sys.path.insert(1,'/Users/sukhkuma/Dev/Programming/Research/MinMax_AA/MinMax_AndersonAcceleration/GANCodebase')

from Model.UntrainedModel.TestGAN import Generator, Discriminator
from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs
from Dataset.LoadNormal10DimData import normal10DimTrainDs, normal10DimTestDs

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Loss function
def DLoss(G,D,z,x):
    return torch.mean(torch.log(D(x)) + torch.log(1-D(G(z))))
    

def GLoss(G,D,z):
    return -torch.mean(torch.log(1-D(G(z))))


# GAN Training Function
def TrainModel(mnistDl,normalDl,G,D,numEpochs):
    dOpt = Adam(
        D.parameters(),
        lr=0.00001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    gOpt = Adam(
        G.parameters(),
        lr=0.00001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    dLosses = []
    gLosses = []
    epochs = []
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch}")
        for (mnistBatch,normalBatch) in zip(mnistDl,normalDl):
            realData = mnistBatch[0].view(-1,28*28)
            
            # Train D
            dOpt.zero_grad()
            dloss_ = DLoss(G,D,normalBatch,realData)
            dloss_.backward()
            dOpt.step()
            dLosses.append(dloss_.item())
            
            # Train G
            gOpt.zero_grad()
            gLoss_ = GLoss(G,D,normalBatch)
            gLoss_.backward()
            gOpt.step()
            gLosses.append(gLoss_.item())
        epochs.append(epoch)
        
    return np.array(epochs), np.array(gLosses), np.array(dLosses)
    

# Init model def
G = Generator()
D = Discriminator()

# Take Ds and convert to Dl with same batchSize across both random and real data
batchSize = 5
mnistTrainDl = DataLoader(mnistTrainDs, batch_size=batchSize, shuffle=True)
mnistTestDl = DataLoader(mnistTestDs, batch_size=batchSize, shuffle=True)
normal10DimTrainDl = DataLoader(normal10DimTrainDs, batch_size=batchSize, shuffle=True)
normal10DimTestDl = DataLoader(normal10DimTestDs, batch_size=batchSize, shuffle=True)

# Train Model
epochs, gLosses, dLosses = TrainModel(mnistTrainDl,normal10DimTrainDl,G,D,numEpochs=2)

print(dLosses[4000:5000])
