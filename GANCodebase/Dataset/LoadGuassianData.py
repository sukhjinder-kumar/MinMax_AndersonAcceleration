'''
Use case:

>>> from Dataset.LoadGuassianDataset import guassian8Ds,guassian8Dl,guassian25Ds,guassian25Dl
>>> samples = []
>>> for i in range(len(guassian25Ds)): # sub 25 by 8 for guassian8Ds
...     sample = guassian25Ds[i].numpy()
...     samples.append(sample)
>>> samples = np.array(samples)
>>> plt.scatter(samples[:,0], samples[:,1])
>>> plt.xlim(-3,3)
>>> plt.ylim(-3,3)
>>> plt.show()
<plot is displayed>
>>> for batch in guassian25Dl:
...     temp = batch
...     break
>>> print(temp.shape)
torch.Size([5, 2])
>>> print(temp)
tensor([[-1.4088, -1.4183],
        [ 1.4136, -1.4115],
        [ 1.4481, -0.0038],
        [ 1.4212,  1.3945],
        [ 0.0020,  1.4163]])
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


def Guassian25():
    '''
    @Params
    batchSize : int : batch size
    @Returns
    dataset : ndarray : array of 2D points(# = 100K) ~ Normal with stdDev = ? and mean one of the
    25 points shown below (with ., including 0, 2's, etc atleast before /stddev)
    (read ~ as -ve)
           y-axis
             |
       .  .  4  .  .
             |
       .  .  2  .  .
             |
    --~4-~2--0--2--4--> x-axis
             |
       .  . -2  .  .
             |
       .  . -4  .  .
             |
    '''
    dataset = []
    numPts = int(100000/25) # Number points in one direction
    for i in range(numPts):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2) * 0.05
                point[0] += 2*x
                point[1] += 2*y 
                dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    np.random.shuffle(dataset)
    dataset /= 2.828  # stdev
    return dataset


def Guassian8():
    '''
    @Params
    batchSize : int : batch size
    @Returns
    dataset : ndarray : array of 2D points(# = 100K) ~ Normal with stdDev = ? and mean one of the
    8 points
    '''
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    sigs = [np.eye(2) * 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,
            np.eye(2)* 1e-2,]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    for _ in range(int(100000/8)):
        for center in range(8):
            mu_, sig_ = centers[center], sigs[center]
            data = np.random.multivariate_normal(mu_, sig_)
            dataset.append(data)
    dataset = np.array(dataset, dtype='float32')
    dataset /= 1.414  # stdev
    return dataset


class DatasetGuassian(Dataset):
    def __init__(self, dataset):
        super(DatasetGuassian, self).__init__()
        self.x = torch.from_numpy(dataset)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx]


guassian8Ds = DatasetGuassian(Guassian8())
guassian8Dl = DataLoader(guassian8Ds, batch_size=5, shuffle=True)
guassian25Ds = DatasetGuassian(Guassian25())
guassian25Dl = DataLoader(guassian25Ds, batch_size=5, shuffle=True)
