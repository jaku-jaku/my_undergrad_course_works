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
import a4_lib


def solve_a4_ex1(
    # USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    TOTA_NUM_EPOCHS :int    = 5,
    LEARNING_RATE   :float  = 0.001,
    BATCH_SIZE      :int    = 1,
    MAX_SAMPLES     :int    = 10, # Default: None => all data
    # const:
    OUT_DIR_E1      :str    = "output/E1",
    IMG_SIZE        :tuple  = (32, 32),
    VERBOSE_LEVEL   :int    = a4_lib.VerboseLevel.HIGH,
    TRAINING_AUG    :List   = [], # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
):
    # INIT: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    ### Directory generation ###
    a4_lib.create_all_folders(DIR=OUT_DIR_E1)

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
    train_dataset = a4_lib.A4_EX1_CNN_HELPER.load_mnist_data(
        batch_size   = BATCH_SIZE, 
        resize       = IMG_SIZE, # NOTE: make sure you understand why
        n_workers    = 1,
        augmentation = TRAINING_AUG, # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
        shuffle      = False,
        train_set    = True,
    )
    test_dataset  = a4_lib.A4_EX1_CNN_HELPER.load_mnist_data(
        batch_size   = BATCH_SIZE, 
        resize       = IMG_SIZE, # NOTE: make sure you understand why
        n_workers    = 1,
        augmentation = None, # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
        shuffle      = True,
        train_set    = False,
    )

    # TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    # train & evaulate:
    report = a4_lib.A4_EX1_CNN_HELPER.train_and_monitor(
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
    t.save(MODEL_DICT["VGG11"].state_dict(), "%s/last.pth"%(OUT_DIR_E1))

    # EVALUATE: --- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    # P4.2
    report.output_progress_plot(
        OUT_DIR         = OUT_DIR_E1,
        tag             = "VGG11_{}".format("_".join(TRAINING_AUG)),
        verbose_level   = VERBOSE_LEVEL
    )
    # P4.3 - test augmented dataset:
    for augmentation in ["HFlip", "VFlip", "GAUSS-0.01", "GAUSS-0.1", "GAUSS-1"]:
        if VERBOSE_LEVEL >= a4_lib.VerboseLevel.LOW:
            print("==== P4.3 : Test: {}".format(augmentation))
        
        test_dataset_aug  = a4_lib.A4_EX1_CNN_HELPER.load_mnist_data(
            batch_size   = BATCH_SIZE, 
            resize       = IMG_SIZE, # NOTE: make sure you understand why
            n_workers    = 1,
            augmentation = augmentation,
            shuffle      = True,
            train_set    = False,
        )

        test_loss, test_acc, test_n, test_ellapse = a4_lib.A4_EX1_CNN_HELPER.test(
            device        = device,
            test_dataset  = test_dataset,
            loss_func     = CrossEntropyLoss(),
            net           = MODEL_DICT["VGG11"], 
            verbose_level = VERBOSE_LEVEL,
            max_data_samples = MAX_SAMPLES,
        )

        # report:
        if VERBOSE_LEVEL >= a4_lib.VerboseLevel.MEDIUM:
            print("> [{}] test_loss: {}, test_acc: {}, test_n: {}, test_ellapse: {}".format(
                augmentation, test_loss, test_acc, test_n, test_ellapse
            ))
