"""
This main would predict likes based on the tweets only
"""
# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from ast import literal_eval
import os
import sys

from dataclasses import dataclass, field
from typing import Dict, Any, List
import re
import emoji
import operator
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import datetime

# ML:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# debugger:
from icecream import ic

import jieba # to split east-asian language to words

## Custom Files:
def abspath(relative_path):
    ABS_PATH = "/home/jx/JXProject/Github/UW__4B_Individual_Works/CS 480/Kaggle"
    return os.path.join(ABS_PATH, relative_path)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(abspath("src_code"))

# Custom Lib
import jx_lib
from jx_pytorch_lib import ProgressReport

# %% LOAD DATASET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# import data
TRAIN_DATA_X = pd.read_csv(abspath("data/p_train_x.csv"))
TRAIN_DATA_Y = pd.read_csv(abspath("data/p_train_y.csv"))
TEST_DATA_X = pd.read_csv(abspath("data/p_test_x.csv"))
ic(np.sum(TRAIN_DATA_X["id"] == TRAIN_DATA_Y["id"])) # so the assumption should be right, they share the exact same id in sequence
TRAIN_DATA = pd.concat([TRAIN_DATA_X, TRAIN_DATA_Y["likes_count"]], axis=1)
ic(TRAIN_DATA.shape)

# %% Pre-Data Analysis: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
ANALYSIS_OUTPUT_FOLDER =  abspath("output-analysis")
jx_lib.create_folder(ANALYSIS_OUTPUT_FOLDER)

TRAIN_DATA.head(10)
sns.displot(TRAIN_DATA["likes_count"])
HEADERS = list(TRAIN_DATA.columns)
a = ic(HEADERS)

# Plot Language and video Count:
fig = plt.figure(figsize=(20,20))
ax = plt.subplot(2, 2, 1)
ax.set_title("Language Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="language", hue="likes_count", multiple="dodge")
fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')

ax = plt.subplot(2, 2, 2)
ax.set_title("Video Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="video", hue="likes_count", multiple="dodge")
fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')


# %% DEFINE NETWORK: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
"""
Logistic Regression Bag-of-Words classifier
"""
class BOW_Module(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOW_Module, self).__init__()
        # the parameters of the affine mapping.
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # input => Linear => softmax
        return F.log_softmax(self.linear(bow_vec), dim=1)

class BOW_ModuleV2(nn.Module):
    def __init__(self, num_labels, vocab_size, 
        dropout=0.2, d_hidden=100, n_layers=2,
    ):
        super(BOW_ModuleV2, self).__init__()
        self.linear1 = nn.Linear(vocab_size, d_hidden)
        self.linear2 = nn.Linear(d_hidden, num_labels)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, bow_vec):
        # input => Linear => softmax
        y = self.linear1(bow_vec)
        # y = self.dropout(y)
        y = self.linear2(y)
        outputs =  F.log_softmax(y, dim=1)
        return outputs

class TwitterLikePredictor:
    # %% generate a word to index:
    word_to_ix = {}
    word_count = {}
    label_to_ix = {}
    NUM_LABELS = 0
    VOCAB_SIZE = 0

    @dataclass
    class PredictorConfiguration:
        Y_TAG                : str              = "likes_count"
        USE_GPU              : bool             = True
        PROCESSED_TWEET_TAG  : str              = "norm-tweet"
        OUTPUT_FOLDER        : str              = abspath("output")
        MODEL_TAG            : str              = "default"
        # tweet pre-processing: 
        DELIMITER_SET        : str              = "; |, |、|。| "
        SYMBOLE_REMOVE_LIST  : str              = field(default_factory=lambda: ["\[", "\]", "\(", "\)"])
        KEYS_TO_REMOVE_LIST  : List             = field(default_factory=lambda: ["http", "arXiv", "https"])
        # training set:
        SHUFFLE_TRAINING     : bool             = False
        PERCENT_TRAINING_SET : float            = 0.9
        # bag of words (Tweeter Interpretation):
        BOW_TOTAL_NUM_EPOCHS : int              = 10
        LOSS_FUNC            : nn               = nn.NLLLoss()
        LEARNING_RATE        : float            = 0.001
        FORCE_REBUILD        : bool             = True 
        OPTIMIZER            : float            = optim.SGD
        MOMENTUM             : float            = 0.9
        MODEL_VERSION        : str              = "v2"
        D_HIDDEN             : int              = 100

    def _print(self, content):
        if self.verbose:
            print("[TLP] ", content)

    def __init__(
        self, 
        pd_data_training,
        verbose,
        config: PredictorConfiguration
    ):
        self.config = config
        self.verbose = verbose
        # gen folder:
        jx_lib.create_folder(self.config.OUTPUT_FOLDER)
        jx_lib.create_folder(abspath("processed_data"))
        # Pre-processing Dataset ==== ==== ==== ==== ==== ==== ==== #
        path = abspath('processed_data/train-[{}].csv'.format(self.config.MODEL_TAG))
        if os.path.exists(path) and not self.config.FORCE_REBUILD:
            self.training_dataset = pd.read_csv(path)
            self._print("Loading Pre-Processed Test Dataset From {} [size:{}]".format(path, self.training_dataset.shape))
        else:
            self._print("Pre-processing Dataset ...")
            self.training_dataset = self.generate_tweet_message_normalized_column(pd_data=pd_data_training, config=self.config)
            # save frame
            self._print("Processed Dataset Saved @ {}".format(path))
            self.training_dataset.to_csv(path)
        # sample:
        UNIQ_LANG =  self.training_dataset["language"].unique().tolist()
        for lang in UNIQ_LANG:
            index =  self.training_dataset.index[ self.training_dataset["language"] == lang].tolist()[0]
            self._print("{} > {}".format(lang,  self.training_dataset["norm-tweet"][index]))
        labels = self.training_dataset[self.config.Y_TAG].unique()
        self.label_to_ix = {i:i for i in labels}
        self.NUM_LABELS = len(labels)
        # Prepare Hardware ==== ==== ==== ==== ==== ==== ==== #
        self._print("Prepare Hardware")
        if self.config.USE_GPU:
            self.device = self.load_device()
        # Prepare Dataset ==== ==== ==== ==== ==== ==== ==== #
        self._print("Prepare Training Dataset")
        self.pytorch_data_train_id, self.pytorch_data_eval_id, \
            self.pytorch_data_train, self.pytorch_data_eval = self.split_training_dataset(
                TRAIN_DATA = self.training_dataset,
                config = self.config
            )
        self._generate_bow_dictionary(
            training_data = self.pytorch_data_train,
            evaluation_data = self.pytorch_data_eval
        )
        # Init Model ==== ==== ==== ==== ==== ==== ==== ==== #
        self._print("New Model Created")
        self.create_new_model(version=self.config.MODEL_VERSION)
        if self.config.USE_GPU:
            self.model.to(self.device)
        # REPORT ==== ==== ==== ==== ==== ==== ==== ==== === #
        self._print("REPORT SUMMARY ======================= ")
        ic(self.VOCAB_SIZE)
        ic(self.NUM_LABELS)
        # sample top bag of words
        self.word_count_top_100 = sorted(self.word_count.items(), key=lambda x:-np.sum(x[1]))[:100]
        # print model parameters
        if self.version in ["v2"]:
            self._print("MODEL: {}".format(self.model))
        else:
            for param in self.model.parameters():
                self._print("Model Shape: ".format(param.shape))
        self._print("======================= END OF INIT =======================")
    
    @staticmethod
    def generate_tweet_message_normalized_column(
        pd_data, 
        config
    ):
        MAX_LENGTH, d = pd_data.shape
        tweet_data = []
        for i in range(MAX_LENGTH):
            messages = pd_data['tweet'][i]
            # remove some characters with space
            for sym in config.SYMBOLE_REMOVE_LIST:
                messages = re.sub(sym, " ", messages)
            # separate delimiter:
            messages = re.split(config.DELIMITER_SET, messages)
            # split emojis
            new_messages = []
            for msg in messages:
                new_messages.extend(emoji.get_emoji_regexp().split(msg))
            messages = new_messages
            # remove keys:
            new_messages = []
            for msg in messages:
                no_key = True
                for key in config.KEYS_TO_REMOVE_LIST: # tags to be removed
                    if key in msg:
                        msg = key # no_key = False, lets replace key with key name => as feature
                if no_key and len(msg) > 0:
                    # split:
                    new_messages.extend(jieba.lcut(msg, cut_all=True)) # split east asian
            
            # Let's convert time and other correlation into word as well!
            time_str = pd_data["created_at"][i]
            time_str = time_str.split(" ")
            date_ = datetime.datetime.strptime(time_str[0], "%Y-%m-%d")
            time_ = datetime.datetime.strptime(time_str[1], "%H:%M:%S")
            descriptive_str = [
                # time related:
                "[year:{}]".format(date_.year),
                "[month:{}]".format(date_.month),
                "[hour:{}]".format(time_.hour),
                "[zone:{}]".format(time_str[2])
            ]
            # existance of other placeholders
            if (not pd.isna(pd_data["place"][i])):
                descriptive_str.append("[exist:place]")
            if (not pd.isna(pd_data["quote_url"][i])):
                descriptive_str.append("[exist:quote_url]")
            if (not pd.isna(pd_data["thumbnail"][i])):
                descriptive_str.append("[exist:thumbnail]")
            if (len(pd_data["reply_to"][i]) > 0):
                descriptive_str.append("[exist:reply:to]")
            # include hashtags: (should already be inside the tweet, but let's emphasize it by repeating)
            descriptive_str.extend(literal_eval(pd_data['hashtags'][i]))
            # extend messages
            new_messages.extend(descriptive_str)
            # append to normalized tweet data
            tweet_data.append(new_messages)

        pd_data['norm-tweet'] = tweet_data
        return pd_data


    def create_new_model(self, version="v2"):
        self._print("Use Model Version: {}".format(version))
        self.version = version
        if version == "v2":
            # self.model = nn.Sequential(
            #     nn.Linear(self.VOCAB_SIZE, self.NUM_LABELS),
            #     nn.Softmax(dim=1)
            # )
            self.model = BOW_ModuleV2(self.NUM_LABELS, self.VOCAB_SIZE, d_hidden=self.config.D_HIDDEN)
        else:
            self.model = BOW_Module(self.NUM_LABELS, self.VOCAB_SIZE)

    @staticmethod
    def load_device():
        # hardware-acceleration
        device = None
        if torch.cuda.is_available():
            print("[ALERT] Attempt to use GPU => CUDA:0")
            device = torch.device("cuda:0")
        else:
            print("[ALERT] GPU not found, use CPU!")
            device =  torch.device("cpu")
        return device
    
    # gen pytorch data:
    @staticmethod
    def pandas2pytorch(
        pd_data,
        x_tag: str,
        y_tag: str,
        range: List[int]
    ) -> "id, training pair":
        id_,x_,y_ = pd_data['id'][range[0]:range[1]], pd_data[x_tag][range[0]:range[1]], pd_data[y_tag][range[0]:range[1]]
        return id_.tolist(), [(x,y) for x,y in zip(x_,y_)]

    @staticmethod
    def split_training_dataset(
        TRAIN_DATA, config
    ):
        # let's shuffle the training data:
        if config.SHUFFLE_TRAINING:
            TRAIN_DATA = TRAIN_DATA.sample(frac = 1)
        
        N_TRAIN = int(len(TRAIN_DATA) * config.PERCENT_TRAINING_SET)
        N_TEST = len(TRAIN_DATA) - N_TRAIN
        # let's split data
        train_id,pytorch_data_train = TwitterLikePredictor.pandas2pytorch(
            pd_data = TRAIN_DATA,
            x_tag = config.PROCESSED_TWEET_TAG, y_tag = config.Y_TAG,
            range =[0, N_TRAIN]
        )
        test_id,pytorch_data_eval = TwitterLikePredictor.pandas2pytorch(
            pd_data = TRAIN_DATA,
            x_tag = config.PROCESSED_TWEET_TAG, y_tag = config.Y_TAG,
            range =[N_TRAIN, N_TRAIN+N_TEST]
        )
        return train_id, test_id, pytorch_data_train, pytorch_data_eval

    def _generate_bow_dictionary(
        self,
        training_data, 
        evaluation_data
    ):
        for sent, y in training_data + evaluation_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.word_count[word] = [0] * self.NUM_LABELS
                else:
                    self.word_count[word][y] += 1
        self.VOCAB_SIZE = len(self.word_to_ix)

    def make_bow_vector(self, sentence):
        vec = torch.zeros(len(self.word_to_ix))
        for word in sentence:
            # do not use word if it was not in the dictionary, this happens when unseen testing dataset
            if word in self.word_to_ix:
                vec[self.word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(self, label):
        return torch.LongTensor([self.label_to_ix[label]])

    def train(self, gen_plot:bool=True, sample_threshold:float=0.5):
        self._print("\n\nTRAING BEGIN -----------------------------:")
        report_ = ProgressReport()
        optimizer_ = self.config.OPTIMIZER(
            self.model.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        loss_ = self.config.LOSS_FUNC

        n_accuracy_drops = 0
        for epoch in range(self.config.BOW_TOTAL_NUM_EPOCHS):
            self._print("> epoch {}/{}:".format(epoch + 1, self.config.BOW_TOTAL_NUM_EPOCHS))
    
            train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
            val_loss_sum, val_acc_sum, val_n, val_start = 0.0, 0.0, 0, time.time()

            # TRAIN -----------------------------:
            i, n = 0, len(self.pytorch_data_train)
            for instance, label in self.pytorch_data_train:
                i += 1
                print("\r > Training [{}/{}]".format(i, n),  end='')
                # 1: Clear PyTorch Cache
                self.model.zero_grad()
                # 2: Convert BOW to vectors:
                bow_vec = self.make_bow_vector(instance)
                target = self.make_target(label)

                # 3: fwd:
                if self.config.USE_GPU:
                    bow_vec = bow_vec.to(self.device)
                    target = target.to(self.device)
                log_probs = self.model(bow_vec)

                # 4: backpropagation (training)
                loss = loss_(log_probs, target)
                loss.backward()
                optimizer_.step()
                # Log summay:
                train_loss_sum += loss.item()
                train_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
                # train_acc_sum += (log_probs.argmax(dim=1) == yi).sum().item()
                train_n += 1
            
            train_ellapse = time.time() - train_start
            print("\n",  end='')
    
            # TEST -----------------------------:
            i, n = 0, len(self.pytorch_data_eval)
            with torch.no_grad(): # Not training!
                for instance, label in self.pytorch_data_eval:
                    i += 1
                    print("\r > Validating [{}/{}]".format(i, n),  end='')
                    bow_vec = self.make_bow_vector(instance)
                    target = self.make_target(label)
                    if self.config.USE_GPU:
                        bow_vec = bow_vec.to(self.device)
                        target = target.to(self.device)
                    log_probs = self.model(bow_vec)
                    # Log summay:
                    loss = loss_(log_probs, target)
                    val_loss_sum += loss.item()
                    val_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
                    val_n += 1
        
            val_ellapse = time.time() - val_start
            print("\n",  end='')

            val_acc = val_acc_sum / val_n
            if epoch > 1 and report_.history["test_acc"][-1] > val_acc:
                n_accuracy_drops += 1
            else:
                n_accuracy_drops = 0

            # Log ------:
            report_.append(
                epoch         = epoch,
                train_loss    = train_loss_sum / train_n,
                train_acc     = train_acc_sum / train_n,
                train_time    = train_ellapse,
                test_loss     = val_loss_sum / val_n,
                test_acc      = val_acc,
                test_time     = val_ellapse,
                learning_rate = 0,
                verbose       = True
            )

            if (val_acc >= sample_threshold): 
                # early sampling, save model that meets minimum threshold
                self._print("> [Minimum Goal Reached] Attempt to predict, with {}>={}:".format(val_acc, sample_threshold))
                tag = "autosave-e:{}".format(epoch)
                pd_data_processed, df_pred = TLP_Engine.predict(pd_data=TEST_DATA_X, tag=tag)
                self.save_model(tag=tag)
            
            if (n_accuracy_drops >= 3):
                self._print("> Early Stopping due to accuracy drops in last 3 iterations!")
                break

        self._print("End of Program")

        # OUTPUT REPORT:
        if gen_plot:
            report_.output_progress_plot(
                figsize       = (15,12),
                OUT_DIR       = self.config.OUTPUT_FOLDER,
                tag           = self.config.MODEL_TAG
            )
        # SAVE Model:
        self.save_model(tag="final")
        return report_
    
    def predict(self, pd_data, tag=None):
        # pre-process:
        self._print("Pre-processing Test Dataset ...")
        path = abspath('processed_data/test-[{}-{}].csv'.format(self.config.MODEL_TAG, tag))
        if os.path.exists(path) and not self.config.FORCE_REBUILD:
            self._print("Loading Pre-Processed Test Dataset From {}".format(path))
            pd_data_processed = pd.read_csv(path)
        else:
            pd_data_processed = self.generate_tweet_message_normalized_column(
                pd_data = pd_data, config = self.config
            )
            self._print("Processed Test Dataset Saved @ {}".format(path))
            self.training_dataset.to_csv(path)
        # prediction:
        self._print("Predicting ...")
        y_pred = []
        with torch.no_grad(): # Not training!
            for x in TEST_DATA_X[self.config.PROCESSED_TWEET_TAG]:
                bow_vec = self.make_bow_vector(x)
                if self.config.USE_GPU:
                    bow_vec = bow_vec.to(self.device)
                log_probs = self.model(bow_vec)
                y_pred.append(log_probs.argmax(dim=1))
        # convert to df_pred
        self._print("Converting to dataframe ...")
        df_pred = pd.DataFrame({'label':[y.tolist()[0] for y in y_pred]})
        # save to file:
        if tag is not None:
            path = abspath('processed_data/test_y_pred-[{}-{}].csv'.format(self.config.MODEL_TAG, tag))
            self._print("Prediction of the Test Dataset Saved @ {}".format(path))
            df_pred.to_csv(path, index_label="id")

        return pd_data_processed, df_pred

    def save_model(self, tag):
        torch.save(self.model.state_dict(), abspath("output/model-{}-{}.pt".format(self.config.MODEL_TAG, tag)))


# USER PARAMS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# Pre-evaluation:
# CONFIG = TwitterLikePredictor.PredictorConfiguration(
#     Y_TAG                 = "likes_count",
#     USE_GPU               = True,
#     PROCESSED_TWEET_TAG   = "norm-tweet",
#     OUTPUT_FOLDER         = abspath("output"),
#     MODEL_TAG             = "test-v2.5-cross",
#     # tweet pre-processing:
#     DELIMITER_SET         = '; |, |、|。| ',
#     SYMBOLE_REMOVE_LIST   = ["\[", "\]", "\(", "\)"],
#     KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
#     # training set:
#     SHUFFLE_TRAINING      = False,
#     PERCENT_TRAINING_SET  = 0.90, # 0.99
#     # bag of words (Tweeter Interpretation):
#     LOSS_FUNC             = nn.NLLLoss(),
#     BOW_TOTAL_NUM_EPOCHS  = 10, # 20
#     LEARNING_RATE         = 0.001,
#     FORCE_REBUILD         = True, # False seems to be slower, I guess caches might be better
#     OPTIMIZER             = optim.SGD,
#     MOMENTUM              = 0.8,
#     MODEL_VERSION         = "v2"
#     # PERCENT_TRAINING_SET  = 0.9,
#     # # Best Model so far: test-epoch-1 (Incorporate time ... info.) => 0.462 acc --> 0.45604
#     # BOW_TOTAL_NUM_EPOCHS  = 10,
#     # LEARNING_RATE         = 0.001,
#     # # Best Model so far: test-epoch-1 => 0.4365 acc --> 0.44792
#     # BOW_TOTAL_NUM_EPOCHS  = 20,
#     # LEARNING_RATE         = 0.001,
#     # # Best Model so far: test-epoch-180
#     # BOW_TOTAL_NUM_EPOCHS  = 180,
#     # LEARNING_RATE         = 0.001,
# )

# # INIT ENGINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# TLP_Engine = TwitterLikePredictor(pd_data_training=TRAIN_DATA, verbose=True, config=CONFIG)

# # TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# report = TLP_Engine.train(gen_plot=True, sample_threshold=0.5)

# # PREDICTION: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# pd_data_processed, df_pred = TLP_Engine.predict(pd_data=TEST_DATA_X, tag="test")

# %% Auto overnight training: ----- ----- ----- ----- ----- ----- ----- -----
DICT_OF_CONFIG = {
    "trial-1": TwitterLikePredictor.PredictorConfiguration(
        # Y_TAG                 = "likes_count",
        # USE_GPU               = True,
        # PROCESSED_TWEET_TAG   = "norm-tweet",
        # OUTPUT_FOLDER         = abspath("output"),
        MODEL_TAG             = "trail-1",
        # tweet pre-processing:
        # DELIMITER_SET         = '; |, |、|。| ',
        # SYMBOLE_REMOVE_LIST   = ["\[", "\]", "\(", "\)"],
        # KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
        # training set:
        # SHUFFLE_TRAINING      = False,
        # PERCENT_TRAINING_SET  = 0.90, # 0.99
        # bag of words (Tweeter Interpretation):
        # LOSS_FUNC             = nn.NLLLoss(),
        # BOW_TOTAL_NUM_EPOCHS  = 10, # 20
        # LEARNING_RATE         = 0.001,
        # FORCE_REBUILD         = True, # False seems to be slower, I guess caches might be better
        # OPTIMIZER             = optim.SGD,
        D_HIDDEN              = 100,
        MOMENTUM              = 0.8,
        MODEL_VERSION         = "v2"
    ),
    "trial-2": TwitterLikePredictor.PredictorConfiguration(
        # Y_TAG                 = "likes_count",
        # USE_GPU               = True,
        # PROCESSED_TWEET_TAG   = "norm-tweet",
        # OUTPUT_FOLDER         = abspath("output"),
        MODEL_TAG             = "trail-2",
        # tweet pre-processing:
        # DELIMITER_SET         = '; |, |、|。| ',
        # SYMBOLE_REMOVE_LIST   = ["\[", "\]", "\(", "\)"],
        # KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
        # training set:
        # SHUFFLE_TRAINING      = False,
        # PERCENT_TRAINING_SET  = 0.90, # 0.99
        # bag of words (Tweeter Interpretation):
        # LOSS_FUNC             = nn.NLLLoss(),
        BOW_TOTAL_NUM_EPOCHS  = 20, # 20
        # LEARNING_RATE         = 0.001,
        # FORCE_REBUILD         = True, # False seems to be slower, I guess caches might be better
        # OPTIMIZER             = optim.SGD,
        D_HIDDEN              = 400,
        MOMENTUM              = 0.8,
        MODEL_VERSION         = "v2"
    ),
    "trial-3": TwitterLikePredictor.PredictorConfiguration(
        # Y_TAG                 = "likes_count",
        # USE_GPU               = True,
        # PROCESSED_TWEET_TAG   = "norm-tweet",
        # OUTPUT_FOLDER         = abspath("output"),
        MODEL_TAG             = "trail-3",
        # tweet pre-processing:
        # DELIMITER_SET         = '; |, |、|。| ',
        # SYMBOLE_REMOVE_LIST   = ["\[", "\]", "\(", "\)"],
        # KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
        # training set:
        # SHUFFLE_TRAINING      = False,
        # PERCENT_TRAINING_SET  = 0.90, # 0.99
        # bag of words (Tweeter Interpretation):
        # LOSS_FUNC             = nn.NLLLoss(),
        BOW_TOTAL_NUM_EPOCHS  = 100, # 20
        LEARNING_RATE         = 0.0001,
        # FORCE_REBUILD         = True, # False seems to be slower, I guess caches might be better
        # OPTIMIZER             = optim.SGD,
        D_HIDDEN              = 400,
        MOMENTUM              = 0.8,
        MODEL_VERSION         = "v2"
    ),
}

min_threshold = 0.47
for name_, config_ in DICT_OF_CONFIG.items():
    print("================================ ================================ BEGIN:{} , Goal:{} =>".format(name_, min_threshold))
    # INIT ENGINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    TLP_Engine = TwitterLikePredictor(pd_data_training=TRAIN_DATA, verbose=True, config=config_)

    # TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    report = TLP_Engine.train(gen_plot=True, sample_threshold=min_threshold)
    max_validation_acc = np.max(report.history["test_acc"])
    if max_validation_acc > min_threshold:
        print("\n>>>> Best Model So Far: {} \n".format(max_validation_acc))
        min_threshold = max_validation_acc # rise standard

    # PREDICTION: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    pd_data_processed, df_pred = TLP_Engine.predict(pd_data=TEST_DATA_X, tag=name_)


# %% Word Analysis: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
list_top100 = list(map(np.array, zip(* TLP_Engine.word_count_top_100)))
# Plot Language and video Count:
fig = plt.figure(figsize=(40,40))
ax = plt.subplot(1, 1, 1)
ax.set_title("Top 100 Repeated Word Count")
plt.bar(list_top100[0], list_top100[1][:, 0], label="0")
plt.bar(list_top100[0], list_top100[1][:, 1], label="1")
plt.bar(list_top100[0], list_top100[1][:, 2], label="2")
plt.bar(list_top100[0], list_top100[1][:, 3], label="3")

# %% POST-ANALYSIS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
"""
We will do post-analysis here, to see the validation performance of the model.
"""
N_TRAIN = int(len(TLP_Engine.training_dataset) * TLP_Engine.config.PERCENT_TRAINING_SET)
N_TEST = len(TLP_Engine.training_dataset) - N_TRAIN
validation_data = pd.DataFrame(data=TLP_Engine.training_dataset, index=range(N_TRAIN,(N_TRAIN+N_TEST)))
# # %% HACK:
# dict_tweet_to_id = {}
# for id_, tweet in zip(validation_data['id'], validation_data['norm-tweet']):
#     dict_tweet_to_id[''.join(tweet)] = id_
#%% Lets try redo the test, and analyze it
n = len(TLP_Engine.pytorch_data_eval)
validation_data["pred-likes"] = [0] * n
validation_data["pred-ifcorrect"] = [False] * n
validation_data["pred-probabilities"] = [[0,0,0,0]] * n
loss_ = TLP_Engine.config.LOSS_FUNC
with torch.no_grad(): # Not training!
    i = -1
    for instance, label in TLP_Engine.pytorch_data_eval:
        i += 1
        print("\r > Predicting [{}/{}]".format(i+1, n),  end='')
        bow_vec = TLP_Engine.make_bow_vector(instance)
        target = TLP_Engine.make_target(label)
        if TLP_Engine.config.USE_GPU:
            bow_vec = bow_vec.to(TLP_Engine.device)
            target = target.to(TLP_Engine.device)
        log_probs = TLP_Engine.model(bow_vec)
        y_pred = log_probs.argmax(dim=1).tolist()[0]
        # Log summay:
        loss = loss_(log_probs, target)

        # find index and store:
        # id_ = dict_tweet_to_id[''.join(instance)]
        id_ = TLP_Engine.pytorch_data_eval_id[i]
        validation_data.loc[id_, "pred-likes"] = (y_pred)
        validation_data.loc[id_, "pred-ifcorrect"] = ((y_pred == label))
        validation_data.loc[id_, "pred-probabilities"] = (loss.cpu().tolist())

validation_data.to_csv(abspath("processed_data/valid-[{}].csv".format(TLP_Engine.config.MODEL_TAG)))

# %%  convert everything useful to quantity
validation_data["time-year"] = [0] * n
validation_data["time-month"] = [0] * n
validation_data["time-date"] = "" * n
validation_data["time-seconds"] = [0] * n
validation_data["time-zone"] = "" * n

validation_data["if-place"] = [False] * n
validation_data["if-quote"] = [False] * n
validation_data["if-thumbnail"] = [False] * n
validation_data["if-reply_to"] = [False] * n
for id_ in validation_data['id']:
    # convert creation time: => time affects how many ppl viewed the post
    time_str = validation_data.loc[id_, "created_at"]
    time_str = time_str.split(" ")
    date_ = datetime.datetime.strptime(time_str[0], "%Y-%m-%d")
    time_ = datetime.datetime.strptime(time_str[1], "%H:%M:%S")

    validation_data.loc[id_, "time-year"] = date_.year
    validation_data.loc[id_, "time-month"] = date_.month
    validation_data.loc[id_, "time-date"] = time_str[0]
    validation_data.loc[id_, "time-seconds"] = (time_ - datetime. datetime(1900, 1, 1)).total_seconds()
    validation_data.loc[id_, "time-zone"] = time_str[2]
    # other:
    validation_data.loc[id_, "if-place"] = not pd.isna(validation_data.loc[id_, "place"])
    validation_data.loc[id_, "if-quote"] = not pd.isna(validation_data.loc[id_, "quote_url"])
    validation_data.loc[id_, "if-thumbnail"] = not pd.isna(validation_data.loc[id_, "thumbnail"])
    validation_data.loc[id_, "if-reply_to"] = len(validation_data.loc[id_,"reply_to"]) > 0

validation_data.to_csv(abspath("processed_data/valid-converted-[{}].csv".format(TLP_Engine.config.MODEL_TAG)))

# %%
fig = plt.figure(figsize=(20,20))

DICT_SUBPLOTS = {
    "Prediction Result (likes_count)": {'x': "likes_count", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Video vs. Correctness": {'x': "video", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(s) vs. Correctness": {'x': "time-seconds", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(zone) vs. Correctness": {'x': "time-zone", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(year) vs. Correctness": {'x': "time-year", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(month) vs. Correctness": {'x': "time-month", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-place vs. Correctness": {'x': "if-place", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-quote vs. Correctness": {'x': "if-quote", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-thumbnail vs. Correctness": {'x': "if-thumbnail", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-reply_to vs. Correctness": {'x': "if-reply_to", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
}
n_plot = np.ceil(np.sqrt(len(DICT_SUBPLOTS)))
i = 0
for title, entry in DICT_SUBPLOTS.items():
    i += 1
    ax = plt.subplot(n_plot, n_plot, i)
    ax.set_title(title)
    sns.histplot(ax=ax, data=validation_data, x=entry["x"], y=entry["y"], hue=entry["hue"], multiple=entry["mult"])

fig.savefig("{}/plot_{}-[{}].png".format(ANALYSIS_OUTPUT_FOLDER, "post-process-summary", TLP_Engine.config.MODEL_TAG), bbox_inches = 'tight')


# %% Confusion Matrix:
cf = confusion_matrix(validation_data["likes_count"], validation_data["pred-likes"])

fig, status = jx_lib.make_confusion_matrix(
    cf=cf,
    group_names=None,
    categories='auto',
    title="Prediction Summary"
)
fig.savefig("{}/plot_{}-conf_mat-[{}].png".format(ANALYSIS_OUTPUT_FOLDER, "post-process-summary", TLP_Engine.config.MODEL_TAG), bbox_inches = 'tight')

# %% Entry Correlation Plot
partial_validation_data = pd.DataFrame(data=validation_data, columns =["time-year", "time-month", "if-thumbnail", "likes_count", "pred-likes"])
corr = partial_validation_data.corr()
fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
fig.savefig("{}/plot_{}-correlation-[{}].png".format(ANALYSIS_OUTPUT_FOLDER, "post-process-summary", TLP_Engine.config.MODEL_TAG), bbox_inches = 'tight')

# %%
