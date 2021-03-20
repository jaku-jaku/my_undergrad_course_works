# python typical
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Optional
from enum import Enum, IntEnum, auto
from dataclasses import dataclass
import os
from datetime import datetime
import operator

# sklearn
from sklearn.model_selection import train_test_split

# tensor flow
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

## SYS HELPER ##
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class P3_Env:
    _train_data_x: List[float]  = None
    _train_data_y: List[float]  = None
    _test_data_x: List[float]   = None
    _test_data_y: List[float]   = None

    @staticmethod
    def print(content: str):
        print("[ P3_Env ] > {}".format(content))

    def __init__(
        self, 
        f_data_function, 
        x_range: List[float],
        env_name: str,
        # common configuration
        data_pts_i: List[int],       
        hidden_nodes_j: List[int],   
        N_eval_per_model: int,
        MAX_DATA_SIZE: int,
        TRAIN_SIZE: float
    )->None:
        # create folders
        mkdir("fig")
        mkdir("fig/p3")
        mkdir("fig/p3/{}".format(env_name))

        # store configs
        self._f_data_function = f_data_function
        self._x_range = x_range
        self._data_pts_i = data_pts_i
        self._hidden_nodes_j = hidden_nodes_j
        self._N_eval_per_model = N_eval_per_model
        self._MAX_DATA_SIZE = MAX_DATA_SIZE
        self._TRAIN_SIZE = TRAIN_SIZE
        self._env_name = env_name

        # generate model:
        self._generate_mlp_models()

        # generate data:
        self._generate_data_set()

        # report
        self.print("Data Size: [ Train: {train} | Test: {test} ] #Models: {n_model}"\
            .format(train=np.shape(self._train_data_x), test=np.shape(self._test_data_x), n_model=len(self._dict_of_mlps)))

    def _generate_data_set(self):
        # generate data
        data_x = np.random.uniform(self._x_range[0], self._x_range[1], self._MAX_DATA_SIZE)
        data_y = self._f_data_function(data_x)

        # split train and test data
        self._train_data_x, self._test_data_x, self._train_data_y, self._test_data_y = \
            train_test_split(data_x, data_y, train_size=self._TRAIN_SIZE, shuffle=True)

    def _generate_mlp_models(self):
        self._dict_of_mlps = {}
        for n_pts in self._data_pts_i:
            for n_nodes in self._hidden_nodes_j:
                tag = "i={}-j={}".format(n_pts, n_nodes)
                self._dict_of_mlps[tag] = {
                    "n_pts": n_pts,
                    "n_nodes": n_nodes,
                    "avg_training_errors": [],
                    "avg_validation_errors": [], 
                    "lowest_training_error": 1.0, 
                    "lowest_validation_error": 1.0, 
                }

    def plot_progress(
        self,
        hs,
        tag:str,
    ):
        plt.figure()
        fig2 = plt.gcf()
        for h in hs:
            plt.plot(np.log10(h.history['loss']), 'b')
            if "val_loss" in h.history:
                plt.plot(np.log10(h.history['val_loss']), 'r')
                plt.ylabel("Loss")
            else:
                plt.ylabel("Training Loss")
            plt.xlabel("epoch")
        if 'val_loss' in h.history:
            plt.legend(["Training", "Validation"])
        fig2.savefig("fig/p3/{env}/train_loss_{tag}.png".format(env=self._env_name,tag=tag), bbox_inches = 'tight')
        plt.close(fig2)

    def plot_fitness_result(self, mlp, tag, mlp_index):
        plt.figure()
        fig2 = plt.gcf()
        
        if mlp_index is not None:
            instance = self._dict_of_mlps[mlp_index]
            tx = self._test_data_x[0:instance["n_pts"]]
            ty = self._test_data_y[0:instance["n_pts"]]
            C = sorted(zip(tx,ty), key=operator.itemgetter(0))
            new_x, new_y = zip(*C)
            x = new_x
            y_true = new_y
        else:
            x = np.linspace(self._x_range[0], self._x_range[1], num=1000)
            y_true = self._f_data_function(x)
        
        y_pred = mlp.predict(x)
        plt.plot(x, y_true, 'b')
        plt.plot(x, y_pred, 'r')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(["Ground Truth", "Prediction"])
        fig2.savefig("fig/p3/{env}/final_fit_{tag}.png".format(env=self._env_name,tag=tag), bbox_inches = 'tight')
        plt.close(fig2)
    
    def plot_and_print_loss_matrix(self):
        entries = ["lowest_training_error","lowest_validation_error"]
        for topic in entries:
            mat = np.zeros((len(self._data_pts_i), len(self._hidden_nodes_j)))
            # extract result & matrify the result
            for i, n_pts in enumerate(self._data_pts_i):
                for j, n_nodes in enumerate(self._hidden_nodes_j):
                    tag = "i={}-j={}".format(n_pts, n_nodes)
                    mat[i][j] = self._dict_of_mlps[tag][topic]
            # print result
            print("== Matrix {} ===".format(topic))
            print(mat)
            # plot result
            plt.figure()
            fig2 = plt.gcf()
            plt.imshow(mat)
            plt.yticks(list(range(len(self._data_pts_i))), self._data_pts_i)
            plt.xticks(list(range(len(self._hidden_nodes_j))), self._hidden_nodes_j)
            plt.ylabel("i (Data Points)")
            plt.xlabel("j (Number of Neurons)")
            plt.colorbar()
            file_name = "fig/p3/{env}/matrix_{tag}.png".format(env=self._env_name,tag=topic)
            print(file_name)
            fig2.savefig(file_name, bbox_inches = 'tight')
            plt.close(fig2)

    def train_at(
        self,
        mlp_index: str,
        N_epoch: int, # Early stopping
    ):
        # train only one instance
        instance = self._dict_of_mlps[mlp_index]
        # reset session:
        keras.backend.clear_session()
        # build mlp
        mlp = keras.models.Sequential([
            Dense(instance["n_nodes"], activation='sigmoid', input_shape=(1,),
                kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=30),
                bias_initializer=keras.initializers.RandomNormal(mean=0., stddev=10)
            ),
            Dense(1, activation='linear')
        ]) 
        mlp.compile(loss='mean_squared_error', optimizer='adam')
        # train the model fully
        h = mlp.fit(
            self._train_data_x[0:instance["n_pts"]],# n_pts training
            self._train_data_y[0:instance["n_pts"]],# n_pts training
            epochs=N_epoch, 
            batch_size=1, 
            verbose=0
        )
        # save model
        mlp.save('fig/p3/{}/best_mlp'.format(self._env_name))
        return h, mlp
    
    def evaluate_test_data(
        self,
        mlp_index: str,
        mlp
    ):
        instance = self._dict_of_mlps[mlp_index]
        tx = self._test_data_x[0:instance["n_pts"]]
        ty = self._test_data_y[0:instance["n_pts"]]
        result = mlp.evaluate(tx, ty)
        return result

    
    def run_hyperParam_optimization(
        self,
        callback_termination,
        k_fold:int,
        N_epoch:int, # Early stopping
        N_trial:int,
        plot:bool
    ):
        min_key = None
        ind = 0
        tot = N_trial * len(self._dict_of_mlps)
        for tag, instance in self._dict_of_mlps.items():
            # data selection
            training_pair = list(zip(self._train_data_x, self._train_data_y))
            # down sample to limited number of data for training
            training_pair = training_pair[0:instance["n_pts"]]

            ### t-TRIAL ========== ========== ==========
            for t in range(N_trial):
                ind += 1
                print("==== TEST [{tag:10s}] : Trial [{t}/{nt}] : [{ind}/{tot}]====".format(
                    tag=tag, t=(t+1), nt=N_trial, ind=ind, tot=tot
                ))
                
                # data shuffling per trial
                np.random.shuffle(training_pair)

                ### K-FOLD  ========== ========== ==========
                # divide data into k-portions
                data_pool = np.array_split(training_pair, k_fold) 

                ### MLP     ========== ========== ==========
                # create storage:
                subfold_memory = {
                    "training_error"        : np.zeros(k_fold),
                    "validation_error"      : np.zeros(k_fold),
                    "training_history"      : [],
                }
                
                ## apply k-fold    ========== ==========
                for kf in range(k_fold):
                    # reset session:
                    keras.backend.clear_session()
                    # build mlp
                    mlp = keras.models.Sequential([
                        Dense(instance["n_nodes"], activation='sigmoid', input_shape=(1,),
                            kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=30),
                            bias_initializer=keras.initializers.RandomNormal(mean=0., stddev=10)
                        ),
                        Dense(1, activation='linear')
                    ]) 
                    mlp.compile(loss='mean_squared_error', optimizer='adam')
                    # construct training set
                    train_set = np.concatenate((data_pool[0:kf] + data_pool[kf+1:k_fold]), axis=0)
                    # construct validation set
                    valid_set = data_pool[kf]
                    # decode training data
                    tx,ty = zip(*train_set)
                    tx,ty = np.array(tx), np.array(ty)
                    # decode validation data
                    vx,vy = zip(*valid_set)
                    vx,vy = np.array(vx), np.array(vy)
                    # train the model
                    h = mlp.fit(
                        tx, 
                        ty, 
                        epochs=N_epoch, 
                        batch_size=1, 
                        verbose=0,
                        validation_data=(vx, vy),
                        callbacks = [callback_termination]
                    )
                    # store:
                    if plot:
                        subfold_memory["training_history"   ].append(h)
                    subfold_memory["training_error"     ] = h.history['loss'][-1]
                    subfold_memory["validation_error"   ] = h.history['val_loss'][-1]
                
                # create plots
                if plot:
                    self.plot_progress(hs=subfold_memory["training_history"  ], tag=tag)
                ## compute final average losses
                instance["avg_training_errors"     ].append(np.average(subfold_memory["training_error"     ]))
                instance["avg_validation_errors"   ].append(np.average(subfold_memory["validation_error"   ]))
    
            # capture the best out of N_trial
            instance["lowest_training_error"]   = min(instance["avg_training_errors"])
            instance["lowest_validation_error"] = min(instance["avg_validation_errors"])

            if (min_key is None) or (self._dict_of_mlps[min_key]["lowest_validation_error"] > instance["lowest_validation_error"]):
                min_key = tag

        return min_key, self._dict_of_mlps[min_key]


def main_env(
    f_func, 
    tag, 
    x_range, 
    plot_progress           :bool,
    min_key                 :Optional[str], # None: 3.activation
    N_EPOCH_HARD_STOP       :int,
):

    ## INIT Environment Engine
    env = P3_Env(
        f_data_function     = f_func,
        x_range             = x_range,
        data_pts_i          = [10,40,80,200],
        hidden_nodes_j      = [2,10,40,100],
        env_name            = tag,
        N_eval_per_model    = 5, # repeat the process 5 times by shuffling the data generated randomly
        MAX_DATA_SIZE       = 500,
        TRAIN_SIZE          = 0.8 # 80 % for training by default
    )

    if min_key is None:
        # Run Engine
        val_err_callback = keras.callbacks.EarlyStopping(monitor='val_loss', baseline=0.001, patience=100, mode="min") 
        min_key, min_instance = env.run_hyperParam_optimization(
            k_fold                  = 10, 
            N_epoch                 = N_EPOCH_HARD_STOP, 
            plot                    = plot_progress, 
            callback_termination    = val_err_callback,
            N_trial                 = 5
        ) 

        ## Print Engine Result
        print("== SUMMARY ==")
        env.plot_and_print_loss_matrix()
        print(env._dict_of_mlps)
        
        print("== OPTIMAL MODEL ==")
        print(min_instance)

    print(min_key)
    ## Retrain the best model
    h, best_mlp_new = env.train_at(
        mlp_index=min_key,
        N_epoch=N_EPOCH_HARD_STOP
    )
    env.plot_fitness_result(mlp=best_mlp_new, tag="Best[Test Data]{}".format(min_key), mlp_index=min_key)
    env.plot_fitness_result(mlp=best_mlp_new, tag="Best[Overall]{}".format(min_key), mlp_index=None)
    env.plot_progress(hs=[h], tag="Best{}".format(min_key))
    ## hard evaluation:
    result = env.evaluate_test_data(mlp=best_mlp_new, mlp_index=min_key)
    print("Final Training Loss: {}".format(h.history['loss'][-1]))
    print("Final Test Loss: {}".format(result))

def main():
    ENABLE_P3_2 = True # Else RUN: P3-1 to generate hyperparam matrices
    ##
    if ENABLE_P3_2:
        min_key_f1 = "i=80-j=100"
        min_key_f2 = "i=40-j=40"
    else:
        min_key_f1 = None
        min_key_f2 = None

    # f1
    main_env(
        f_func              = (lambda x: x * np.sin(6 * np.pi * x) * np.exp(- x ** 2)),
        x_range             = [-1, 1],
        tag                 = "F1",
        plot_progress       = False,
        min_key             = min_key_f1,#None, # None : for auto-tuning
        N_EPOCH_HARD_STOP   = 1000,
    )
    # f2
    main_env(
        f_func              = (lambda x: np.exp(- x ** 2) * np.arctan(x) * np.sin(4 * np.pi * x)),
        x_range             = [-2, 2],
        tag                 = "F2",
        plot_progress       = False,
        min_key             = min_key_f2, # None : for auto-tuning
        N_EPOCH_HARD_STOP   = 1000,
    )

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("=== END of P3 [ Time Elapse: {} ] ===".format(str(end_time-start_time)))

