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
3. Delete readme files in all the output directories (data, logs, models, results)
Your directory tree should look like this:
```
Aurora-Substorm-Prediction
├── npz
├── src
├── README.md
└── requirements.txt
```

# Data preparation
For training and testing, we use npz files instead of oringinal jpg images. You can download all the npz files ([training](https://github.com/SusanJJN/Aurora-Substrom-Prediction/releases/download/v1.0/training_set.rar), [test](https://github.com/SusanJJN/Aurora-Substorm-Prediction/releases/download/v1.0/test_set.rar) here. After downloading, unzip the .rar and put the training set in /npz/train and put the test set in /npz/test. 

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
The code will finish training and testing on all the bootstrap sampling sets. 

# Generate final prediction
```python
cd src/
python aggregate.py
```

