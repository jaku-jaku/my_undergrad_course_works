import os

import time
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum, auto
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch as t
import torchvision.transforms as ttf
from torchvision.datasets import MNIST


def get_files(DIR:str, file_end:str=".png"):
    return [ os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(file_end) ]

def create_all_folders(DIR:str):
    path_ = ""
    for folder_name_ in DIR.split("/"):
        path_ = os.path.join(path_, folder_name_)
        create_folder(path_, False)

def clean_folder(DIR:str):
    create_folder(DIR=DIR, clean=True)

def create_folder(DIR:str, clean:bool=False):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    elif clean:
        filelist = get_files(DIR)
        for f in filelist:
            os.remove(f)

@dataclass
class ProgressReport:
    history: Dict

    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_time": [],
            "test_loss": [],
            "test_acc": [],
            "test_time": [],
            "learning_rate": [],
        }

    
    def append(
        self,
        epoch,
        train_loss,
        train_acc,
        train_time,
        test_loss,
        test_acc,
        test_time,
        learning_rate,
        verbose = True,
    ):
        self.history["epoch"]         .append(epoch        )
        self.history["train_loss"]    .append(train_loss   )
        self.history["train_acc"]     .append(train_acc    )
        self.history["train_time"]    .append(train_time   )
        self.history["test_loss"]     .append(test_loss    )
        self.history["test_acc"]      .append(test_acc     )
        self.history["test_time"]     .append(test_time    )
        self.history["learning_rate"] .append(learning_rate)
        if verbose:
            print('    epoch {} > Training: [LOSS: {:.4f} | ACC: {:.4f}] | Testing: [LOSS: {:.4f} | ACC: {:.4f}] Ellapsed: {:.2f}+{:.2f} s | rate:{:.5f}'.format(
                epoch + 1, train_loss, train_acc, test_loss, test_acc, train_time, test_time, learning_rate
            ))


    def output_progress_plot(
        self,
        figsize = (15,12),
        OUT_DIR = "",
        tag     = ""
    ):
        xs = self.history['epoch']
        # Plot
        fig = plt.figure(figsize=figsize)
        plt.subplot(2, 1, 1)
        plt.plot(xs, self.history['train_acc'], label="training")
        plt.plot(xs, self.history['test_acc'], label="testing")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.xticks(xs)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(xs, self.history['train_loss'], label="training")
        plt.plot(xs, self.history['test_loss'], label="testing")
        plt.xticks(xs)
        plt.ylabel("Loss (cross-entropy)")
        plt.xlabel("epoch")
        plt.legend()
        fig.savefig("{}/training_progress[{}].png".format(OUT_DIR, tag), bbox_inches = 'tight')
        plt.close(fig)

        return fig
        
class VerboseLevel(IntEnum):
    NONE    = auto()
    LOW     = auto()
    MEDIUM  = auto()
    HIGH    = auto()

################################
######## EX 1 : Helper #########
################################
class A4_EX1_CNN_HELPER:
    # LOAD DATASET: --- ----- ----- ----- ----- ----- ----- ----- ----- #
    # Definition:
    @staticmethod
    def load_mnist_data(
        batch_size: int, 
        resize    : Optional[tuple] = None,
        n_workers : int  = 1,
        root      : str = "./data/",
    ):
        print("=== Loading Data ... ")
        trans = []

        # Image augmentation
        if resize:
            trans.append(ttf.Resize(size=resize))
        trans.append(ttf.RandomHorizontalFlip())
        trans.append(ttf.ToTensor())
        transform = ttf.Compose(trans)
        
        data_train = MNIST(root=root, train=True, download=True, transform=transform)
        data_test = MNIST(root=root, train=False, download=True, transform=transform)
        
        train_dataset = t.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        test_dataset = t.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        print("=== Loading Data [x] ===")
        return train_dataset, test_dataset

    # TRAINING: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    @staticmethod
    def train(
        train_dataset, 
        test_dataset, 
        optimizer, 
        loss_func,
        net, 
        num_epochs: int,
        # history_epoch_resolution: float = 1.0, TODO: mini-batches progress!!!
        max_data_samples: Optional[int] = None,
        verbose_level: VerboseLevel = VerboseLevel.LOW,
    ):
        report = ProgressReport()
        # Cross entropy
        for epoch in range(num_epochs):
            if verbose_level >= VerboseLevel.LOW:
                print("> epoch {}/{}:".format(epoch + 1, num_epochs))
            
            # Training:
            if verbose_level >= VerboseLevel.LOW:
                print("  >> Learning (wip)")
            train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
            for i, (X, y) in enumerate(train_dataset):
                if max_data_samples is not None:
                    if i >= max_data_samples:
                        break
                    if verbose_level >= VerboseLevel.HIGH:
                        print("   >[{}/{}]".format(i, max_data_samples), end='\r')
                elif verbose_level >= VerboseLevel.HIGH:
                    print("   >[{}/{}]".format(i, len(train_dataset)),  end='\r')
                # Predict:
                y_prediction = net(X)
                # Calculate loss
                loss = loss_func(y_prediction, y)
                # Gradient descent > [ LEARNING ]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Compute Accuracy
                train_loss_sum += loss.item()
                train_acc_sum += (y_prediction.argmax(dim=1) == y).sum().item()
                train_n += y.shape[0]
            
            # Testing:
            if verbose_level >= VerboseLevel.LOW:
                print("  >> Testing (wip)")
            test_loss_sum, test_acc_sum, test_n, test_start = 0.0, 0.0, 0, time.time()
            for i, (X, y) in enumerate(test_dataset):
                if max_data_samples is not None:
                    if i >= max_data_samples:
                        break
                    if verbose_level >= VerboseLevel.HIGH:
                        print("   >[{}/{}]".format(i, max_data_samples),  end='\r')
                elif verbose_level >= VerboseLevel.HIGH:
                    print("   >[{}/{}]".format(i, len(test_dataset)),  end='\r')
                # Predict:
                y_prediction = net(X)
                # Calculate loss
                loss = loss_func(y_prediction, y)
                # Compute Accuracy
                test_loss_sum += loss.item()
                test_acc_sum += (y_prediction.argmax(dim=1) == y).sum().item()
                test_n += y.shape[0]

            # Store
            report.append(
                epoch         = epoch,
                train_loss    = train_loss_sum / train_n,
                train_acc     = train_acc_sum / train_n,
                train_time    = train_start - test_start,
                test_loss     = test_loss_sum / test_n,
                test_acc      = test_acc_sum / test_n,
                test_time     = test_start - time.time(),
                learning_rate = optimizer.param_groups[0]["lr"],
                verbose       = (verbose_level >= VerboseLevel.MEDIUM)
            )
        return report
        