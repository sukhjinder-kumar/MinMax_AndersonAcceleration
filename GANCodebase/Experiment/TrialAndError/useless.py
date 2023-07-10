# Boilerplate for each experiment file. To allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath

# Import Module
# from Model.UntrainedModel.DCGAN import Generator, Discriminator, weights_init
# from Dataset.LoadMnistDataset import mnistTrainDs, mnistTestDs
# from Dataset.LoadCifarDataset import cifarTrainDs,cifarTestDs,cifarTrainDl,cifarTestDl, classes
# from Model.UntrainedModel.TestNN import TestNN
# from Dataset.LoadNormalData import DatasetNormal 
# from Dataset.LoadGuassianData import guassian8Ds,guassian8Dl,guassian25Ds,guassian25Dl
# from Model.UntrainedModel.GuassianGAN import Generator, Discriminator 


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


##############
# Verify DCGAN
##############

# Init model
# dim = 100 
# ngpu = 0
# G = Generator(dim=dim, ngpu=ngpu).apply(weights_init)
# D = Discriminator(ngpu=ngpu).apply(weights_init)
# 
# # Take Ds and convert to Dl with same batchSize across both random and real data
# batchSize = 128 
# mnistTrainDl = DataLoader(mnistTrainDs, batch_size=batchSize, shuffle=True)
# normalDTrainDs = DatasetNormal(dim=dim)
# normalGTrainDs = DatasetNormal(dim=dim)
# normalDTrainDl = DataLoader(normalDTrainDs, batch_size=batchSize, shuffle=True)
# normalGTrainDl = DataLoader(normalGTrainDs, batch_size=batchSize, shuffle=True)
# 
# a = normalDTrainDs[0]
# for batch in normalDTrainDl:
#     temp = batch
#     break
# b = mnistTrainDs[0][0].reshape(28,28)
# for batch in mnistTrainDl:
#     temp2 = batch[0].reshape(-1,28,28)
#     break
# 
# # For G
# # print(a.shape)
# # print(temp.shape)
# # print(G(a).shape)
# # print(G(temp).shape)
# 
# # For D
# print(b.shape)
# print(temp2.shape)
# print(D(b).shape)
# print(D(temp2).shape)


#########################
# Verify LoadCifarDataset
#########################

# a = cifarTrainDs[4]
# img = np.transpose(a[0].numpy(), (1, 2, 0))
# print(img.shape)
# print(a[1])
# plt.imshow(img/2 + 0.5)
# plt.show()
# print(classes[np.argmax(a[1])])
# for batch in cifarTrainDl:
#     temp = batch
#     break
# print(temp[0].shape)
# print(temp[1].shape)


############################
# Verify LoadGuassianDataset
############################

# samples = []
# for i in range(len(guassian25Ds)):
#     sample = guassian25Ds[i].numpy()
#     samples.append(sample)
# 
# samples = np.array(samples)
# plt.scatter(samples[:,0], samples[:,1])
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.show()
# 
# for batch in guassian25Dl:
#     temp = batch
#     break
# print(temp.shape)
# print(temp)


#####################
# Verify GuassianGAN
#####################

# batchSize = 128 
# dim = 8
# G = Generator(dim=dim)
# D = Discriminator()
# 
# guassian8Dl = DataLoader(guassian8Ds, batch_size=batchSize, shuffle=True)
# normalDTrainDs = DatasetNormal(dim=dim, numSample=100000)
# normalDTrainDl = DataLoader(normalDTrainDs, batch_size=batchSize, shuffle=True)
# 
# print(normalDTrainDs[0].shape)
# print(G(normalDTrainDs[0]))
# print(D(guassian8Ds[0]))
# print(D(guassian8Ds[0]).shape)
# 
# for batch in normalDTrainDl:
#     temp = batch
#     break
# print(G(temp).shape)
# 
# for batch in guassian8Dl:
#     temp = batch
#     break
# print(temp.shape)
# print(D(temp).shape)
