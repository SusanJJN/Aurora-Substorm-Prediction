# Aurora-Substorm-Prediction
This is the implementation of A ConvLSTM-based Prediction Model of Aurora Evolution during the Substorm Expansion Phase

# Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed.

# Requirements
* Python 3.6
* CUDA 9.0
* PyTorch 1.8.0

# Installation
1. Clone this repo:
```python
git clone https://github.com/SusanJJN/Aurora-Subsotrm-Prediction.git
```
2. Install dependencies:
```python
pip install requirements.txt
```
3. Your directory tree should look like this:
```
Aurora-Substorm-Prediction
├── npz
├── results
├── src
    ├── data
    ├── model
        ├── ConvLSTM
    ├── util
├── README.md
└── requirements.txt
```

# Data preparation
For training and testing, we use npz files instead of oringinal jpg images. Put the training set in /npz/train and put the test set in /npz/test. 
You can download the [dataset](https://github.com/SusanJJN/Aurora-Substorm-Prediction/releases/download/v1.0/Dataset.rar).

# Generate bootstrap sampling sets
```python
cd src
python bootstrap.py
```

# Training and Testing
```python
cd src/model/ConvLSTM
python runMix.py
```
The code will finish the training processes on all the bootstrap sampling sets and generate the testing results. The testing results are saved in the long_test file in each training log file, and the checkpoint.pth is the saved best model file.

# Generate final prediction
```python
cd src
python aggregate.py
```

A pretrained model can be downloaded, which was trained using the whole training set. This model is provided for testing.
'''python
cd src/model/ConvLSTM
python runTest.py
'''
