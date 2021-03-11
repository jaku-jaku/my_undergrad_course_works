# python
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum, auto

# sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #,KFold,cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# debug:
from icecream import ic

# jx-lib
import jx_lib

class YLabel(IntEnum):
    airplane   = 0
    automobile = auto()
    bird       = auto()
    cat        = auto()
    deer       = auto()
    dog        = auto()
    frog       = auto()
    horse      = auto()
    ship       = auto()
    truck      = auto()

def main():
    # USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    ### MODEL ###
    model_list = {
        "MLP": Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(512, activation='sigmoid'),
            Dense(10, activation='softmax')
        ]),
        "CNN-1": Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            Flatten(),
            Dense(512, activation='sigmoid'),
            Dense(10, activation='softmax')
        ]),
        "CNN-2": Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='sigmoid'),
            Dropout(0.2),
            Dense(512, activation='sigmoid'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ]),
    }
    ### CONST ###
    PERCENT_TRAINING_SET = 0.2
    BATCH_SIZE           = 32
    MAX_EPOCHS           = 5
    LEARNING_RATE        = 0.001
    LOSS_METHOD          = 'categorical_crossentropy'
    METRIC               = 'accuracy'
    # INIT: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    # MODES_AVALIABLE = ["unmodified", "balance"]
    # if mode not in MODES_AVALIABLE:
    #     raise ValueError("Invalid mode selection!!")
    ### Directory generation ###
    OUT_DIR = "output/p3"#{}".format(mode)
    jx_lib.create_all_folders(DIR=OUT_DIR)
    #     # directory cleaning
    #     jx_lib.clean_folder(DIR=OUT_DIR)
    def file_path(file_name, tag=".png"):
        return "{}/{}.{}".format(OUT_DIR, file_name, tag)
    # DATA: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    ### IMPORT DATA ###
    (X_train_original, y_train_original), (X_test_original, y_test_original) = cifar10.load_data()
    # sample test images for visual reference:
    sample_imgs = {}
    for label in YLabel:
        index = np.where(y_test_original == label)
        sample_imgs[label.name] = X_test_original[index[0][0]]/255.0 # normalize too
    ### PRE-PROCESSING DATA ###
    # one-hot encoding:
    y_train, y_test = to_categorical(y_train_original), to_categorical(y_test_original)
    # normalization (min-max) , since we know the image data is in [0,255]:
    X_train, X_test = X_train_original/255.0, X_test_original/255.0
    # randomly sample 20% of the training set as the training set:
    n_trainingset = len(y_train)
    downsample_index_test_data = np.random.randint(0,n_trainingset,int(n_trainingset * PERCENT_TRAINING_SET))
    X_train = X_train[downsample_index_test_data]
    y_train = y_train[downsample_index_test_data]
    # dataset:
    ic(np.shape(y_train))
    ic(np.shape(X_train))
    ic(np.shape(y_test))
    ic(np.shape(X_test))
    # output sample dataset images:
    fig = jx_lib.imgs_plot(dict_of_imgs=sample_imgs)
    fig.savefig(file_path("sample_imgs"), bbox_inches = 'tight')
    plt.close(fig)
    # TRAIN: ---- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    # Print summary:
    for model_name, model in model_list.items():
        ic(model.summary())

    histories = {}
    for model_name, model in model_list.items():
        model.compile(
            optimizer   = Adam(lr=LEARNING_RATE), 
            loss        = LOSS_METHOD,
            metrics     = [METRIC]
        )
        histories[model_name] = model.fit(
            X_train, y_train, 
            verbose=1, 
            batch_size=BATCH_SIZE, 
            epochs=MAX_EPOCHS,
            validation_data=(X_test, y_test)
        )

    # SUMMARY: - ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    for h_name, h in histories.items():
        # report
        ic(h.history['accuracy'][-1])
        ic(h.history['val_accuracy'][-1])
        ic(h.history['loss'][-1])
        ic(h.history['val_loss'][-1])
        # plot
        fig = jx_lib.progress_plot(h=h)
        fig.savefig(file_path("progress_{}".format(h_name)), bbox_inches = 'tight')
        plt.close(fig)

    # SAMPLE: - ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    for model_name, model in model_list.items():
        labels = []
        dict_input_x = {}
        dict_y_pred = {}
        dict_prob = {}
        for label in YLabel:
            labels.append(label.name)
            prediction = model.predict(sample_imgs[label.name].reshape(1, 32, 32, 3))
            probability = np.squeeze(prediction)
            dict_y_pred[label.name] = prediction
            dict_prob[label.name] = probability
        # plot sample results
        jx_lib.output_prediction_result_plot(
            labels       = labels,
            dict_input_x = sample_imgs,
            dict_prob    = dict_prob,
            figsize      = (10, 4),
            OUT_DIR      = OUT_DIR,
            tag          = model_name
        )
        

            
if __name__ == "__main__":
    main()
