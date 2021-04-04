# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import a4_lib
import a4_ex1_solution

import numpy as np
import matplotlib.pyplot as plt
import torch as t

def main():
    # %%
    NUM_EPOCH       = 5
    LEARNING_RATE   = 0.001
    BATCH_SIZE      = 100
    MAX_SAMPLES     = None # Default: None => all data
    VERBOSE_LEVEL   = a4_lib.VerboseLevel.HIGH


    # %%
    # Raw Trial:
    model1 = a4_ex1_solution.solve_a4_ex1(
        # USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
        TOTA_NUM_EPOCHS = NUM_EPOCH,
        LEARNING_RATE   = LEARNING_RATE,
        BATCH_SIZE      = BATCH_SIZE,
        MAX_SAMPLES     = MAX_SAMPLES, # Default: None => all data
        # const:
        OUT_DIR_E1      = "output/E1",
        IMG_SIZE        = (32, 32),
        VERBOSE_LEVEL   = a4_lib.VerboseLevel.HIGH,
        TRAINING_AUG    = [], # Options: ["HFlip", "VFlip", "GAUSS-0.01"],
    )

    # %%
    # Augmented Trial
    model2 = a4_ex1_solution.solve_a4_ex1(
        # USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
        TOTA_NUM_EPOCHS = NUM_EPOCH,
        LEARNING_RATE   = LEARNING_RATE,
        BATCH_SIZE      = BATCH_SIZE,
        MAX_SAMPLES     = MAX_SAMPLES, # Default: None => all data
        # const:
        OUT_DIR_E1      = "output/E1",
        IMG_SIZE        = (32, 32),
        VERBOSE_LEVEL   = a4_lib.VerboseLevel.HIGH,
        TRAINING_AUG    = ["HFlip", "VFlip", "GAUSS-0p5-0p5"],
    )


    # %%
    device = None
    if t.cuda.is_available():
        print("[ALERT] Attempt to use GPU => CUDA:0")
        device = t.device("cuda:0")
    else:
        print("[ALERT] GPU not found, use CPU!")
        device = t.device("cpu")

    for augmentation in ["", "HFlip-1", "VFlip-1", "GAUSS-0.01", "GAUSS-0.1", "GAUSS-1"]:
        # Loading training dataset:
        train_dataset = a4_lib.A4_EX1_CNN_HELPER.load_mnist_data(
            batch_size   = 1, 
            resize       = (32,32), # NOTE: make sure you understand why
            n_workers    = 1,
            augmentation = augmentation,
            shuffle      = False,
            train_set    = False,
        )
        sample_x = None
        sample_y = None
        for i, (X, y) in enumerate(train_dataset):
            if i > 10:
                break

            sample_x = X
            sample_y = y
            fig = plt.figure(figsize=(12,12))
            plt.imshow(sample_x[0][0], "gray")
            fig.savefig("{}/sample_[{}:{}].png".format("output/E1", augmentation, i), bbox_inches = 'tight')
            plt.close(fig)

            # test
            sample_x = sample_x.to(device)
            y_prediction_1 = model1(sample_x).argmax(dim=1)
            sample_x = sample_x.to(device)
            y_prediction_2 = model2(sample_x).argmax(dim=1)
            
            print("[{}]> y:{} | y1:{} , y2:{}".format(augmentation, sample_y, y_prediction_1, y_prediction_2))


# %%
if __name__ == '__main__':
    main()



