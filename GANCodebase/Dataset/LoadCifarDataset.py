'''
Use case:
>>> from Dataset.LoadCifarDataset import cifarTrainDs,cifarTestDs,cifarTrainDl,cifarTestDl,classes
Files already downloaded and verified
Files already downloaded and verified
>>> # Import appropriate libraries
>>> a = cifarTrainDs[0]
>>> a[0].shape
(3,32,32)
>>> img = np.transpose(a[0].numpy(), (1, 2, 0)) # torchvision uses channelxHxW, numpy uses HxWxchannel
>>> img.shape
(32, 32, 3)
>>> print(a[1])
tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
>>> plt.imshow(img/2 + 0.5) # Unnormalize, otherwise value clipping and bad image
<matplotlib.image.AxesImage object at 0x140c4e790>
>>> plt.show()
<image is displayed>
>>> print(classes[np.argmax(a[1])])
frog
>>> for batch in cifarTrainDl:
...     temp = batch
...     break
>>> temp[0].shape
torch.Size([5, 3, 32, 32])
>>> temp[1].shape
torch.Size([5, 10])
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

cifarPath = "./Dataset/DatasetFiles/CIFAR"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

targetTransform = transforms.Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one hot encoding

cifarTrainDs = torchvision.datasets.CIFAR10(
    root=cifarPath, train=True, download=True, transform=transform, target_transform=targetTransform)
cifarTrainDl = torch.utils.data.DataLoader(
    cifarTrainDs, batch_size=5, shuffle=True)
cifarTestDs = torchvision.datasets.CIFAR10(
    root=cifarPath, train=False, download=True, transform=transform, target_transform=targetTransform)
cifarTestDl = torch.utils.data.DataLoader(
    cifarTestDs, batch_size=5, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
