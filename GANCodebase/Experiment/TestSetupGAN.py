# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

# Import Module
from Model.UntrainedModel.TestGAN import Generator, Discriminator
from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs
from Dataset.LoadNormal10DimData import normal10DimTrainDs, normal10DimTestDs

# Import Libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math


# Discriminator Loss Function
def DLoss(G,D,z,x):
    return -torch.mean(torch.log(D(x)) + torch.log(1-D(G(z))))
    

# Generator Loss Function
def GLoss(G,D,z):
    return torch.mean(torch.log(1-D(G(z))))


# GAN Training Function
def TrainModel(mnistDl,normalDl,G,D,numEpochs):
    dOpt = Adam(
        D.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    gOpt = Adam(
        G.parameters(),
        lr=0.0001,
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
            dLoss_ = DLoss(G,D,normalBatch,realData)
            dLoss_.backward()
            dOpt.step()
            dLosses.append(dLoss_.item())
            
            # Train G
            gOpt.zero_grad()
            gLoss_ = GLoss(G,D,normalBatch)
            gLoss_.backward()
            gOpt.step()
            gLosses.append(gLoss_.item())

        epochs.append(epoch)
        # Check if nan occured
        if math.isnan(gLosses[-1]) or math.isnan(dLosses[-1]):
            break
        
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
epochs, gLosses, dLosses = TrainModel(mnistTrainDl,normal10DimTrainDl,G,D,numEpochs=1)

# Viz
fakeImage = G(normal10DimTrainDs[0]).reshape(28,28)
plt.imshow(fakeImage.detach().numpy())
saveFigPath = "./Figure/Experiment/TestSetupGAN/GeneratedImage.png"
plt.savefig(saveFigPath) # Save viz
plt.show()

# Save model
saveGeneratorModelPath = "./Model/TrainedModel/TestGAN/Generator.pt"
saveDiscriminatorModelPath = "./Model/TrainedModel/TestGAN/Discriminator.pt"
torch.save(G.state_dict(), saveGeneratorModelPath)
torch.save(D.state_dict(), saveDiscriminatorModelPath)

# Save loss function in log.txt file
file = open('./Experiment/log.txt','w')
np.set_printoptions(threshold=np.inf)
file.writelines("gLosses")
file.writelines(str(gLosses))
file.writelines("dLosses")
file.writelines(str(dLosses))
file.close()
