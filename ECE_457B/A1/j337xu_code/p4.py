# python typical
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Optional
from enum import Enum, IntEnum, auto
from dataclasses import dataclass

# tensor flow
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

# python debugger
from icecream import ic # Debugger

######################
### DATA LABEL DEF ###
######################
class Constituents(IntEnum):
    Ethanol                       = 1
    Malic_acid                    = 2 
    Ash                           = 3
    Alcalinity_of_ash             = 4 
    Magnesium                     = 5
    Total_phenols                 = 6
    Flavanoids                    = 7
    Nonflavanoid_phenols          = 8
    Proanthocyanins               = 9
    Color_intensity               = 10
    Hue                           = 11
    OD280_OD315_of_diluted_liquid = 12
    Proline                       = 13

class Product(IntEnum):
    P1 = 1
    P2 = 2
    P3 = 3   

class DataType(Enum):
    TRAIN_X = "train_x"
    TRAIN_Y = "train_y"
    TEST_X = "test_x"
    TEST_Y = "test_y"


class P4_ENV:
    def __init__(
        self, 
        DATA: List[List[float]],
        training_percent: float = 0.75, # 75% of data used for training
    ):
        self._data_extraction_and_preparation(
            DATA=DATA,
            Eta_train=training_percent
        )

    def _data_extraction_and_preparation(
        self,
        DATA: List[List[float]],
        Eta_train: float,
    ):
        # find normalization bound
        DATA_MIN_MAX = []
        for entry in Constituents:
            DATA_MIN_MAX.append([min(DATA[:, entry]), max(DATA[:, entry])])
        DATA_MIN_MAX = np.array(DATA_MIN_MAX)

        # categorize by product class 
        DATA_SET = {
            DataType.TRAIN_X  : [],
            DataType.TRAIN_Y  : [],
            DataType.TEST_X   : [],
            DataType.TEST_Y   : []
        }

        for prod in Product:
            # dataset categorization
            data = (DATA[DATA[:,0] == prod, 1:(len(Constituents)+1)])
            # dataset normalization
            data_norm = (data - DATA_MIN_MAX[:,0])/(DATA_MIN_MAX[:,1] - DATA_MIN_MAX[:,0])
            # dataset divide 
            n, m = np.shape(data_norm)
            n_train = round(n * Eta_train)
            # gen labels [ k , # Products ]
            Y_label = np.zeros((n, len(Product)))
            Y_label[:, prod - 1] = 1
            # store inputs
            DATA_SET[DataType.TRAIN_X].append( data_norm[0:n_train , :] )
            DATA_SET[DataType.TEST_X ].append( data_norm[n_train:n , :] )
            # store label
            DATA_SET[DataType.TRAIN_Y].append( Y_label[0:n_train , :] )
            DATA_SET[DataType.TEST_Y ].append( Y_label[n_train:n , :] )

        DATA_SET[DataType.TRAIN_X] = np.concatenate(DATA_SET[DataType.TRAIN_X])
        DATA_SET[DataType.TRAIN_Y] = np.concatenate(DATA_SET[DataType.TRAIN_Y])
        DATA_SET[DataType.TEST_X ] = np.concatenate(DATA_SET[DataType.TEST_X ])
        DATA_SET[DataType.TEST_Y ] = np.concatenate(DATA_SET[DataType.TEST_Y ])

        # store
        self.DATA_MIN_MAX = DATA_MIN_MAX
        self.DATA_SET = DATA_SET

        # check data shape
        ic(np.shape(self.DATA_SET[DataType.TRAIN_X]))
        ic(np.shape(self.DATA_SET[DataType.TRAIN_Y]))
        ic(np.shape(self.DATA_SET[DataType.TEST_X]))
        ic(np.shape(self.DATA_SET[DataType.TEST_Y]))

    def normalize_and_predict(
        self,
        mlp,
        data_x
    ):
        data_x = np.array(data_x)
        data_norm = (data_x - self.DATA_MIN_MAX[:,0])/(self.DATA_MIN_MAX[:,1] - self.DATA_MIN_MAX[:,0])
        print("Normalized: {}".format(data_norm))
        predict_array = mlp.predict([data_norm])
        predict_label = predict_array.argmax() + 1
        return predict_array, predict_label

    def train_all_mlps(
        self,
        dict_of_mlp,
    ):
        """ P4.1 Hyper Tuning Process
        """
        best_instance = None
        best_tag = None
        for tag, instance in dict_of_mlp.items():
            print("==== TEST [{tag:10s}] ====".format(tag=tag))
            instance["mlp"].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            h = instance["mlp"].fit(
                self.DATA_SET[DataType.TRAIN_X], 
                self.DATA_SET[DataType.TRAIN_Y], 
                epochs=instance["max_epoch"], 
                batch_size=20, 
                verbose=0
            )
            train_accuracy = self.plot_progress(h=h, tag=tag)
            test_accuracy = self.validate(self.DATA_SET[DataType.TEST_X],self.DATA_SET[DataType.TEST_Y], mlp=instance["mlp"])

            # record back the result
            instance["train_accuracy"] = train_accuracy
            instance["test_accuracy"] = test_accuracy

            # record best
            if best_instance is None:
                best_instance = instance
                best_tag = tag
            elif best_instance["test_accuracy"] < test_accuracy:
                best_instance = instance
                best_tag = tag
        
        return best_instance, best_tag
    
    def test_run(
        self
    ):
        """ Implementation Validation Code
        """
        mlp = keras.models.Sequential([
            Dense(10, activation='sigmoid', input_shape=(13,)),
            Dense(20, activation='sigmoid'),
            Dense(3, activation='softmax')
        ])

        print(mlp.summary())
        mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        h = mlp.fit(
            self.DATA_SET[DataType.TRAIN_X], 
            self.DATA_SET[DataType.TRAIN_Y], 
            epochs=250, 
            batch_size=20, 
            verbose=0
        )

        self.plot_progress(h=h, tag="test")
        self.validate(self.DATA_SET[DataType.TEST_X],self.DATA_SET[DataType.TEST_Y], mlp=mlp)

    def validate(
        self,
        test_x,
        test_y,
        mlp
    ):
        test_accuracy = 100*mlp.evaluate(test_x, test_y, verbose=0)[1]
        print('Test accuracy: {test_acc:.2f} %'.format(test_acc=test_accuracy))
        return test_accuracy


    def plot_progress(
        self,
        h,
        tag:str
    ):
        train_acc = h.history['accuracy'][-1]*100
        print('Train accuracy: {train_acc:.2f} %'.format(train_acc=train_acc))
        # Plot
        plt.figure()
        plt.plot(h.history['accuracy'])
        plt.ylabel("training accuracy")
        plt.xlabel("epoch")
        fig2 = plt.gcf()
        fig2.savefig("fig/p4/train_accu_{tag}.png".format(tag=tag), bbox_inches = 'tight')

        plt.figure()
        plt.plot(np.log10(h.history['loss']))
        plt.ylabel("training loss")
        plt.xlabel("epoch")

        fig2 = plt.gcf()
        fig2.savefig("fig/p4/train_loss_{tag}.png".format(tag=tag), bbox_inches = 'tight')
        return train_acc

def main():
    ### IMPORT DATA ###
    DATA = np.loadtxt(open("randomized_data.txt"), delimiter=",")
    env = P4_ENV(DATA=DATA)

    # construct mlp test models
    MLP_DICT = {
        "t1": {
            "mlp": keras.models.Sequential([
                Dense(10, activation='sigmoid', input_shape=(13,)),
                Dense(20, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        },
        "t2": {
            "mlp": keras.models.Sequential([
                Dense(10, activation='sigmoid', input_shape=(13,)),
                Dense(20, activation='sigmoid'),
                Dense(20, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        },
        "t3": {
            "mlp": keras.models.Sequential([
                Dense(10, activation='sigmoid', input_shape=(13,)),
                Dense(20, activation='sigmoid'),
                Dense(20, activation='sigmoid'),
                Dense(20, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        },
        "t4": {
            "mlp": keras.models.Sequential([
                Dense(5, activation='sigmoid', input_shape=(13,)),
                Dense(5, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        },
        "t5": {
            "mlp": keras.models.Sequential([
                Dense(5, activation='sigmoid', input_shape=(13,)),
                Dense(5, activation='sigmoid'),
                Dense(5, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        },
        "t6": {
            "mlp": keras.models.Sequential([
                Dense(5, activation='sigmoid', input_shape=(13,)),
                Dense(15, activation='sigmoid'),
                Dense(15, activation='sigmoid'),
                Dense(3, activation='softmax')
            ]),
            "max_epoch": 250,
        }
    }
    
    # Perform Training
    print("\n=== P4.1 ===")
    best_mlp, best_tag = env.train_all_mlps(dict_of_mlp=MLP_DICT)

    # print result
    print("\n=== Summary ===")
    print(MLP_DICT)

    print("\n=== BEST ===")
    print(best_tag)
    print(best_mlp)

    # Perform 4.2 evaluation
    print("\n=== P4.2 ===")
    TEST_DATA = {   "test_a": [13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285],
                    "test_b": [12.04, 4.3, 2.38, 22, 80, 2.1, 1.75, 0.42, 1.35, 2.6, 0.79, 2.57, 580],
                    "test_c": [14.13, 4.1, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560]     }
    for tag, data in TEST_DATA.items():
        print(data)
        predict_array, predict_class = env.normalize_and_predict(mlp=best_mlp["mlp"], data_x=[data])
        print("[{tag:10s}]: Predicted ranking array: {parray}, Classified as: {pclass}".format(tag=tag, parray=predict_array, pclass=predict_class))

if __name__ == "__main__":
    main()
