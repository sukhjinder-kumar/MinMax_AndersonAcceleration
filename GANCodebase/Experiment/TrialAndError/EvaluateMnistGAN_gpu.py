# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

import torch
import matplotlib.pyplot as plt
from Model.UntrainedModel.MnistGAN import Generator
from Dataset.LoadNormalData import DatasetNormal

generatorPath = 'Model/TrainedModel/MnistGAN/Generator_dim100_batchSize128_numEpochs200.pt'
G = Generator()
G.load_state_dict(torch.load(generatorPath))
normal100DimDs = DatasetNormal(100)

for k in range(30):
    plt.imshow(G(normal100DimDs[k]).view(28,28).detach().numpy(),cmap='gray') 
    plt.show()
