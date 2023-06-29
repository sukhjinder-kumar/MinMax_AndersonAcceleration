# MinMax GAN Codebase

## Installation 

`ML` is virtual environment. To install git clone the repo. Create virtual env using `python -m vevn <envName>`, followed by `source <envName>/bin/activate` to use that environement (use `deactivate` to close that environment). To install dependencies for this repo do: `pip install requirements.txt`. You are all set to go!

## Overview

```txt
./
| - ML
    | - ....
| - algorithms.py
| - loadDatasets.py
| - models.py
| - DatasetFiles
    | - MNIST
        | - training.pt
        | - test.pt
| - experimentTest.py
| - README.md
| - TODO.md
```

Names of file and directories are self explanatory. Algo goes in `algorithms.py` file, torch model defintion goes in `models.py`, dataset handling goes in `loadDatasets.py`, raw data goes in `DatasetFiles`.

All files named `experiment<>.py` are experiment files like comparing ADAM and say SGD
