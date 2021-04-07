# python
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Dict

# pytorch:
import torch as t
import torchvision.transforms as ttf
from torch.optim import Adam
from torch.nn import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Dropout,
    Flatten,
    Linear,
    Module,
    CrossEntropyLoss
)

# Custom lib:
import jx_pytorch_lib as jp


# %% [markdown]
# ## E3 - Q1: NN on MNIST > 90%
# %% E3q1
# USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
TOTA_NUM_EPOCHS = 5
LEARNING_RATE   = 0.001
BATCH_SIZE      = 100
MAX_SAMPLES     = None # Default: None => all data
# const:
OUT_DIR_E3      = "output/E3"
IMG_SIZE        = (32, 32)
VERBOSE_LEVEL   = jp.VerboseLevel.HIGH
DATA_AUG        = None #["HFlip", "VFlip", "GAUSS-0p5-0p5"]

# INIT: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
### Directory generation ###
jp.create_all_folders(DIR=OUT_DIR_E3)

### MODEL ###
MODEL_DICT = {
    "VGG11": 
        Sequential(
            ## CNN Feature Extraction
            Conv2d(  1,  64, 3, 1, 1), BatchNorm2d( 64), ReLU(), MaxPool2d(2,2),
            Conv2d( 64, 128, 3, 1, 1), BatchNorm2d(128), ReLU(), MaxPool2d(2,2),
            Conv2d(128, 256, 3, 1, 1), BatchNorm2d(256), ReLU(),
            Conv2d(256, 256, 3, 1, 1), BatchNorm2d(256), ReLU(), MaxPool2d(2,2),
            Conv2d(256, 512, 3, 1, 1), BatchNorm2d(512), ReLU(),
            Conv2d(512, 512, 3, 1, 1), BatchNorm2d(512), ReLU(), MaxPool2d(2,2),
            Conv2d(512, 512, 3, 1, 1), BatchNorm2d(512), ReLU(),
            Conv2d(512, 512, 3, 1, 1), BatchNorm2d(512), ReLU(), MaxPool2d(2,2),
            # Classifier
            Flatten(1),
            Linear( 512, 4096), ReLU(), Dropout(0.5),
            Linear(4096, 4096), ReLU(), Dropout(0.5),
            Linear(4096,   10),
        ),
}

# check device:
# hardware-acceleration
device = None
if t.cuda.is_available():
    print("[ALERT] Attempt to use GPU => CUDA:0")
    device = t.device("cuda:0")
else:
    print("[ALERT] GPU not found, use CPU!")
    device =  t.device("cpu")
MODEL_DICT["VGG11"].to(device)

# Loading training dataset:
train_dataset = jp.A4_EX1_CNN_HELPER.load_mnist_data(
    batch_size   = BATCH_SIZE, 
    resize       = IMG_SIZE, # NOTE: make sure you understand why
    n_workers    = 1,
    augmentation = DATA_AUG, # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
    shuffle      = True,
    train_set    = True,
)
test_dataset  = jp.A4_EX1_CNN_HELPER.load_mnist_data(
    batch_size   = BATCH_SIZE, 
    resize       = IMG_SIZE, # NOTE: make sure you understand why
    n_workers    = 1,
    augmentation = DATA_AUG, # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
    shuffle      = False,
    train_set    = False,
)

# TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# train & evaulate:
report = jp.A4_EX1_CNN_HELPER.train_and_monitor(
    device        = device,
    train_dataset = train_dataset, 
    test_dataset  = test_dataset,
    loss_func     = CrossEntropyLoss(),
    net           = MODEL_DICT["VGG11"], 
    optimizer     = Adam(MODEL_DICT["VGG11"].parameters(), lr=LEARNING_RATE), 
    num_epochs    = TOTA_NUM_EPOCHS,
    verbose_level = VERBOSE_LEVEL,
    max_data_samples = MAX_SAMPLES,
)

# output state:
t.save(MODEL_DICT["VGG11"], "{}/last_{}.pt".format(OUT_DIR_E3, "_".join(TRAINING_AUG)))

