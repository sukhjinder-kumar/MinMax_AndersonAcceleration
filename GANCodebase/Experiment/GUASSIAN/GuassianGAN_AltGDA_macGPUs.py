# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

# Import Module
from Model.UntrainedModel.GuassianGAN import Generator, Discriminator
from Dataset.LoadGuassianData import guassian8Ds
from Dataset.LoadNormalData import DatasetNormal 
from Utils.CreateFile import CreateFile
from Algorithm.AltGDA import AltGDA

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

# For GPU acceleration on MacOS 12.3+, Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    mpsDevice = torch.device("mps")


# Discriminator Loss Function
def DLoss(G,D,z,x):
    # return -torch.mean(torch.log(D(x)) + torch.log(1-D(G(z))))
    BCELoss = nn.BCELoss().to(mpsDevice)
    ones = torch.ones(x.shape[0]).to(mpsDevice)
    zeros = torch.zeros(x.shape[0]).to(mpsDevice)
    dRealLoss = BCELoss(D(x),ones)
    dFakeLoss = BCELoss(D(G(z)),zeros)
    return dRealLoss + dFakeLoss
    

# Generator Loss Function
def GLoss(G,D,z):
    # return torch.mean(torch.log(1-D(G(z))))
    # return -torch.mean(torch.log(D(G(z))))
    BCELoss = nn.BCELoss().to(mpsDevice)
    ones = torch.ones(z.shape[0]).to(mpsDevice)
    gLoss = BCELoss(D(G(z)),ones)
    return gLoss


# GAN Training Function
def TrainModel(guassianDl,normalDDl,normalGDl,G,D,numEpochs):
    opt = AltGDA([
        {'params':D.parameters(), 'lr':0.0002},
        {'params':G.parameters(), 'lr':0.0002}])

    dLosses = []
    gLosses = []
    epochs = []
    N = len(guassianDl)
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch}")
        for i, (guassianBatch,normalDBatch,normalGBatch) in enumerate(zip(guassianDl,normalDDl,normalGDl)):
            realData = guassianBatch.to(mpsDevice)
            normalDBatch = normalDBatch.to(mpsDevice)
            normalGBatch = normalGBatch.to(mpsDevice)
            
            # Train D and G
            opt.zero_grad()
            dLoss = DLoss(G,D,normalDBatch,realData)
            gLoss = GLoss(G,D,normalGBatch)
            opt.step(dLoss,gLoss)
            dLosses.append(dLoss.item())
            gLosses.append(gLoss.item())

            # Like percentage epoch completed
            epochs.append(epoch+i/N)

        epochs.append(epoch+1)

        # Check if nan occured
        if math.isnan(gLosses[-1]) or math.isnan(dLosses[-1]):
            break
        
    return np.array(epochs), np.array(gLosses), np.array(dLosses)
    

# Init model def
dim = 8 
G = Generator(dim=dim).to(mpsDevice)
D = Discriminator().to(mpsDevice)

# Take Ds and convert to Dl with same batchSize across both random and real data
batchSize = 128 
guassian8Dl = DataLoader(guassian8Ds, batch_size=batchSize, shuffle=True)
normalDTrainDs = DatasetNormal(dim=dim, numSample=100000)
normalGTrainDs = DatasetNormal(dim=dim, numSample=100000)
normalDTrainDl = DataLoader(normalDTrainDs, batch_size=batchSize, shuffle=True)
normalGTrainDl = DataLoader(normalGTrainDs, batch_size=batchSize, shuffle=True)

# Train Model
numEpochs = 500
epochs, gLosses, dLosses = TrainModel(guassian8Dl,normalDTrainDl,normalGTrainDl,G,D,numEpochs=numEpochs)
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
saveGeneratorModelPath = "./Model/TrainedModel/GuassianGAN/AltGDA/Generator" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".pt"
saveDiscriminatorModelPath = "./Model/TrainedModel/GuassianGAN/AltGDA/Discriminator" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".pt"
CreateFile(saveGeneratorModelPath)
CreateFile(saveDiscriminatorModelPath)
torch.save(G.state_dict(), saveGeneratorModelPath)
torch.save(D.state_dict(), saveDiscriminatorModelPath)

# Viz: Scatter plot
samples = []
for i in range(1000):
    sample = G(normalGTrainDs[i].to(mpsDevice)).cpu().detach().numpy()
    samples.append(sample)
samples = np.array(samples)
plt.scatter(samples[:,0], samples[:,1])
#plt.xlim(-3,3)
#plt.ylim(-3,3)
saveFigPath = "./Figure/Experiment/GUASSIAN/GuassianGAN_AltGDA/ScatterPlot" + "_dim" \
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
saveFigPath = "./Figure/Experiment/GUASSIAN/GuassianGAN_AltGDA/EpochLossPlot" + "_dim" \
    + str(dim) + "_batchSize" + str(batchSize) + "_numEpochs" + str(numEpochs) + ".png"
CreateFile(saveFigPath)
plt.savefig(saveFigPath) # Save viz
plt.show()

# Printing total time code took
minTook = math.floor((endTime-startTime)/60)
secTook = int((endTime-startTime)%60)
print("Training took", minTook, "min", secTook, "sec to run")
