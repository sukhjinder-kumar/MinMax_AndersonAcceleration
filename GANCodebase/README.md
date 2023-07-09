# MinMax GAN Codebase

Table of Content -
<>

## Installation 

To install git clone the repo. Create a virtual env using `python -m vevn <envName>`, followed by `source <envName>/bin/activate` to use that environement (use `deactivate` to close that environment). To install dependencies for this repo do: `pip install -r requirements.txt`. You are all set to go!

Update `.env` file. Change value of `basePath` to path of `GANCode/` on your machine. Every path is relative to this path. 

**Note**

In general to test anything you should run your code from this base directory. If you choose to run from other directory care must to be taken. In our codebase all executables are in Experiment folder and each file starts with some boilerplate to ensure that. One must add basePath to `sys.path` and change working directory to `basePath` to ensure relative path works nicely across files. Following is the boilerplate code -

```python
# Boilerplate to allow for module calling and messy relative path issues
import sys
import os
from dotenv import load_dotenv
load_dotenv()
basePath = os.getenv('basePath')
sys.path.insert(1,basePath) # For modules
os.chdir(basePath) # Every relative path now from basePath
```

now you can call any module by simply `import Dataset.LoadMnistDataset` or use path for files relative to this basePath without having fear of messing relative paths when called from different file.

## Overview

```txt
.
└── GANCode/
    ├── Dataset/
    │   ├── DatasetFiles/
    │   ├── LoadMnistDataset.py
    │   └── ...
    ├── Model/
    │   ├── TrainedModel/
    │   │   └── TestNN.pt
    │   └── UntrainedModel/
    │       ├── TestNN.py
    │       └── ...
    ├── Algorithm/
    │   ├── algorithm1.py
    │   └── ...
    ├── Experiment/
    │   ├── TestSetupNN.py
    │   ├── .env
    │   └── ...
    ├── Figure/
    │   ├── Experiment/
    │   │   ├── TestSetupNN/
    │   │   │   └── EpochLossPlot.png
    │   │   └── ...
    │   └── ...
    ├── README.md
    ├── MLEnv/
    └── requirements.py
```

Names of file and directories are self explanatory. Algorithm goes in `Algorithm/<AlgoName>.py` file, torch model defintion goes in `Model/UntrainedModel/<ModelName>.py`, trained model stored in its counter part, dataset handling goes in `Dataset/Load<DatasetName>.py`, whereas raw data goes in `DatasetFiles`. Figures goes in `Figure` folder. All the testing, experiments like finding inception score goes in `Experiment` folder.

@TODO:

- Define format for each Directory, like what all should be exported especially in algorithms, datasets, and models

- Add verify using TestSetup files.

- Docs to each file like algorithms

- Update githubPytorchfile
