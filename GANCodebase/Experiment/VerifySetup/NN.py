# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

# Import Modules
from Model.UntrainedModel.TestNN import TestNN
from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs, mnistTrainDl, mnistTestDl
from Utils.CreateFile import CreateFile

# Import libraries
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np


def TrainModel(dl, f, num_epochs):
    opt = SGD(f.parameters(), lr=0.001)
    L = nn.CrossEntropyLoss()
    losses = []
    epochs = []
    N = len(dl)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, (x,y) in enumerate(dl):
            opt.zero_grad()
            loss = L(f(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            epochs.append(epoch+i/N) # Like percentage epoch completed
    return np.array(epochs), np.array(losses)


f = TestNN()
numEpochs = 2
epochs, losses = TrainModel(mnistTrainDl, f, num_epochs=numEpochs)

# Viz
epochs = epochs.reshape(2,-1).mean(axis=1) # along row or 2nd axis (1)
losses = losses.reshape(2,-1).mean(axis=1)
plt.plot(epochs,losses,'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Avg Cross Entropy across dataset')
saveFigPath = "./Figure/Experiment/VerifySetup/NN/EpochLossPlot" \
    + "_batchSize" + str(5) + "_numEpochs" + str(numEpochs) + ".png"
CreateFile(saveFigPath) # Check if file exist or else create
plt.savefig(saveFigPath) # Save viz
plt.show()

# Accuracy
correct = 0
for x,y in mnistTestDs:
    if y.argmax() == f(x).argmax():
        correct += 1
print(f"Test Accuracy: {correct/len(mnistTestDs)*100}")

# Save model
saveModelPath = "./Model/TrainedModel/TestNN/TestNN" \
    + "_batchSize" + str(5) + "_numEpochs" + str(numEpochs) + ".pt"
CreateFile(saveModelPath) # Check if file exist or else create
torch.save(f.state_dict(), saveModelPath)