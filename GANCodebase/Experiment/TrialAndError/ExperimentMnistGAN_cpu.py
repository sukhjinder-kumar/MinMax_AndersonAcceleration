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
from Dataset.LoadNormalData import DatasetNormal 
from Utils.CreateFile import CreateFile

# Import Libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import itertools
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
def TrainModel(mnistDl,normalDDl,normalGDl,G,D,numEpochs):
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
        for i, (mnistBatch,normalDBatch,normalGBatch) in enumerate(zip(mnistDl,normalDDl,normalGDl)):
            realData = mnistBatch[0].view(-1,28*28)
            normalDBatch = normalDBatch
            normalGBatch = normalGBatch
            
            # Train D
            dOpt.zero_grad()
            dLoss_ = DLoss(G,D,normalDBatch,realData)
            dLoss_.backward()
            dOpt.step()
            dLosses.append(dLoss_.item())
            
            # Train G
            gOpt.zero_grad()
            gLoss_ = GLoss(G,D,normalGBatch)
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
normal100DimDTrainDs = DatasetNormal(dim=dim)
normal100DimGTrainDs = DatasetNormal(dim=dim)
normal100DimDTrainDl = DataLoader(normal100DimDTrainDs, batch_size=batchSize, shuffle=True)
normal100DimGTrainDl = DataLoader(normal100DimGTrainDs, batch_size=batchSize, shuffle=True)


# Train Model
numEpochs = 1
epochs, gLosses, dLosses = TrainModel(mnistTrainDl,normal100DimDTrainDl,normal100DimGTrainDl,G,D,numEpochs=numEpochs)
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
saveGeneratorModelPath = "./Model/TrainedModel/MnistGAN/Generator" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".pt"
saveDiscriminatorModelPath = "./Model/TrainedModel/MnistGAN/Discriminator" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".pt"
CreateFile(saveGeneratorModelPath)
CreateFile(saveDiscriminatorModelPath)
torch.save(G.state_dict(), saveGeneratorModelPath)
torch.save(D.state_dict(), saveDiscriminatorModelPath)

# Viz: Generated Image (25 at a time)
size_figure_grid = 5
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)
for k in range(5*5):
    i = k // 5
    j = k % 5
    ax[i, j].cla()
    ax[i, j].imshow(G(normal100DimDTrainDs[k]).detach().view(28, 28).numpy(), cmap='Greys')
saveFigPath = "./Figure/Experiment/TrialAndError/ExperimentMnistGAN/GeneratedImage_5x5" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".png"
CreateFile(saveFigPath)
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
saveFigPath = "./Figure/Experiment/TrialAndError/ExperimentMnistGAN/EpochLossPlot" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".png"
CreateFile(saveFigPath)
plt.savefig(saveFigPath) # Save viz
plt.show()

# Printing total time code took
minTook = math.floor((endTime-startTime)/60)
secTook = int((endTime-startTime)%60)
print("Training took", minTook, "min", secTook, "sec to run")
