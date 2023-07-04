# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

# Import Module
from Model.UntrainedModel.MnistGAN import Generator, Discriminator
from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs
from Dataset.LoadNormal10DimData import DatasetNormal 

# Import Libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# Storing start time to see how long did code run for
startTime = time.time()


# Discriminator Loss Function
def DLoss(G,D,z,x):
    # return -torch.mean(torch.log(D(x)) + torch.log(1-D(G(z))))
    BCELoss = nn.BCELoss()
    ones = torch.ones(x.shape[0])
    zeros = torch.zeros(x.shape[0])
    dRealLoss = BCELoss(D(x),ones)
    dFakeLoss = BCELoss(D(G(z)),zeros)
    return dRealLoss + dFakeLoss
    

# Generator Loss Function
def GLoss(G,D,z):
    # return torch.mean(torch.log(1-D(G(z))))
    # return -torch.mean(torch.log(D(G(z))))
    BCELoss = nn.BCELoss()
    ones = torch.ones(z.shape[0])
    gLoss = BCELoss(D(G(z)),ones)
    return gLoss


# GAN Training Function
def TrainModel(mnistDl,normalDl,G,D,numEpochs):
    dOpt = Adam(
        D.parameters(),
        lr=0.0002,
    )
    gOpt = Adam(
        G.parameters(),
        lr=0.0002,
    )

    dLosses = []
    gLosses = []
    epochs = []
    N = len(mnistDl)
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch}")
        for i, (mnistBatch,normalBatch) in enumerate(zip(mnistDl,normalDl)):
            realData = mnistBatch[0].view(-1,28*28)
            normalBatch = normalBatch
            
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

            # Like percentage epoch completed
            epochs.append(epoch+i/N)

        epochs.append(epoch+1)

        # Check if nan occured
        if math.isnan(gLosses[-1]) or math.isnan(dLosses[-1]):
            break
        
    return np.array(epochs), np.array(gLosses), np.array(dLosses)
    

# Init model def
dim = 100 
G = Generator(dim=dim)
D = Discriminator()

# Take Ds and convert to Dl with same batchSize across both random and real data
batchSize = 128 
mnistTrainDl = DataLoader(mnistTrainDs, batch_size=batchSize, shuffle=True)
normal100DimTrainDs = DatasetNormal(dim=dim)
normal100DimTrainDl = DataLoader(normal100DimTrainDs, batch_size=batchSize, shuffle=True)

# Train Model
numEpochs = 1
epochs, gLosses, dLosses = TrainModel(mnistTrainDl,normal100DimTrainDl,G,D,numEpochs=numEpochs)
endTime = time.time()

# Save loss function in log.txt file
file = open('./Experiment/log.txt','w')
np.set_printoptions(threshold=np.inf)
file.writelines("gLosses")
file.writelines(str(gLosses))
file.writelines("dLosses")
file.writelines(str(dLosses))
file.close()

# Save model
saveGeneratorModelPath = "./Model/TrainedModel/MnistGAN/Generator.pt"
saveDiscriminatorModelPath = "./Model/TrainedModel/MnistGAN/Discriminator.pt"
torch.save(G.state_dict(), saveGeneratorModelPath)
torch.save(D.state_dict(), saveDiscriminatorModelPath)

# Viz: Generated Image
fakeImage = G(normal100DimTrainDs[0]).reshape(28,28)
plt.imshow(fakeImage.detach().numpy(), cmap='gray')
saveFigPath = "./Figure/Experiment/ExperimentMnistGAN/GeneratedImage.png"
plt.savefig(saveFigPath) # Save viz
plt.show()

# Viz: epoch vs Losses
completedEpochs = int(epochs[-1])
epochs = epochs.reshape(completedEpochs,-1).mean(axis=1) # along row or 2nd axis (1)
gLosses = gLosses.reshape(completedEpochs,-1).mean(axis=1)
dLosses = dLosses.reshape(completedEpochs,-1).mean(axis=1)
plt.plot(epochs,gLosses,'o--',label='gLosses',color='blue')
plt.plot(epochs,dLosses,'o--',label='dLosses',color='green')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.title('Avg loss across dataset')
saveFigPath = "./Figure/Experiment/ExperimentMnistGAN/EpochLossPlot.png"
plt.savefig(saveFigPath) # Save viz
plt.show()

# Printing total time code took
print("Training took", endTime - startTime, "to run")
