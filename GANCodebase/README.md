# MinMax GAN Codebase

Table of Content -
<>

## Installation 

To install git clone the repo. Create a virtual env using `python -m vevn <envName>`, followed by `source <envName>/bin/activate` to use that environement (use `deactivate` to close that environment). To install dependencies for this repo do: `pip install -r requirements.txt`. You are all set to go!

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

@TODO: Define format for each Directory, like what all should be exported especially in algorithms, datasets, and models
