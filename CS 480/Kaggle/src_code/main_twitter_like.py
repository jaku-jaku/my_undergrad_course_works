"""
This main would predict likes based on the tweets only
"""
# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from dataclasses import dataclass
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
        USE_GPU              : bool             = False
        PROCESSED_TWEET_TAG  : str              = "norm-tweet"
        OUTPUT_FOLDER        : str              = abspath("output")
        MODEL_TAG            : str              = "test-1"
        # tweet pre-processing: 
        DELIMITER_SET        : str              = "; |, |、|。| "
        SYMBOLE_REMOVE_LIST  : str              = None
        KEYS_TO_REMOVE_LIST  : List             = None
        # training set:
        SHUFFLE_TRAINING     : bool             = False
        PERCENT_TRAINING_SET : float            = 0.9
        # bag of words (Tweeter Interpretation):
        BOW_TOTAL_NUM_EPOCHS : int              = 10
        LOSS_FUNC            : nn               = nn.NLLLoss()
        LEARNING_RATE        : float            = 0.1
        
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
        self._print("Pre-processing Dataset ...")
        self.training_dataset = self.generate_tweet_message_normalized_column(pd_data=pd_data_training, config=self.config)
        # save frame
        path = abspath('processed_data/train-[{}].csv'.format(self.config.MODEL_TAG))
        self._print("Processed Dataset Saved @ {}".format(path))
        self.training_dataset.to_csv(path)
        # sample:
        UNIQ_LANG = TRAIN_DATA["language"].unique().tolist()
        for lang in UNIQ_LANG:
            index = TRAIN_DATA.index[TRAIN_DATA["language"] == lang].tolist()[0]
            self._print("{} > {}".format(lang, TRAIN_DATA["norm-tweet"][index]))
        labels = self.training_dataset[self.config.Y_TAG].unique()
        self.label_to_ix = {i:i for i in labels}
        self.NUM_LABELS = len(labels)
        # Prepare Hardware ==== ==== ==== ==== ==== ==== ==== #
        self._print("Prepare Hardware")
        if self.config.USE_GPU:
            self.device = self.load_device()
        # Prepare Dataset ==== ==== ==== ==== ==== ==== ==== #
        self._print("Prepare Training Dataset")
        self.pytorch_data_train_id, self.pytorch_data_test_id, \
            self.pytorch_data_train, self.pytorch_data_test = self.split_training_dataset(
                TRAIN_DATA = self.training_dataset,
                config = self.config
            )
        self._generate_bow_dictionary(
            training_data = self.pytorch_data_train,
            evaluation_data = self.pytorch_data_test
        )
        # Init Model ==== ==== ==== ==== ==== ==== ==== ==== #
        self._print("New Model Created")
        self.create_new_model()
        if self.config.USE_GPU:
            self.model.to(self.device)
        # REPORT ==== ==== ==== ==== ==== ==== ==== ==== === #
        self._print("REPORT SUMMARY ======================= ")
        ic(self.VOCAB_SIZE)
        ic(self.NUM_LABELS)
        # sample top bag of words
        self.word_count_top_100 = sorted(self.word_count.items(), key=lambda x:-np.sum(x[1]))[:100]
        # print model parameters
        for param in self.model.parameters():
            ic(param)
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
            descriptive_str.extend([item for item in pd_data['hashtags'][i]])
            # extend messages
            new_messages.extend(descriptive_str)
            # append to normalized tweet data
            tweet_data.append(new_messages)

        pd_data['norm-tweet'] = tweet_data
        return pd_data


    def create_new_model(self):
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
        test_id,pytorch_data_test = TwitterLikePredictor.pandas2pytorch(
            pd_data = TRAIN_DATA,
            x_tag = config.PROCESSED_TWEET_TAG, y_tag = config.Y_TAG,
            range =[N_TRAIN, N_TRAIN+N_TEST]
        )
        return train_id, test_id, pytorch_data_train, pytorch_data_test

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

    @staticmethod
    def make_bow_vector(sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            # do not use word if it was not in the dictionary, this happens when unseen testing dataset
            if word in word_to_ix:
                vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    @staticmethod
    def make_target(label, label_to_ix):
        return torch.LongTensor([label_to_ix[label]])

    def train(self, gen_plot:bool=True):
        report_ = ProgressReport()
        optimizer_ = optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        loss_ = self.config.LOSS_FUNC

        for epoch in range(self.config.BOW_TOTAL_NUM_EPOCHS):
            print("> epoch {}/{}:".format(epoch + 1, self.config.BOW_TOTAL_NUM_EPOCHS))
    
            train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
            test_loss_sum, test_acc_sum, test_n, test_start = 0.0, 0.0, 0, time.time()

            # TRAIN -----------------------------:
            i, n = 0, len(self.pytorch_data_train)
            for instance, label in self.pytorch_data_train:
                i += 1
                print(" > Training [{}/{}]".format(i, n),  end='\r')
                # 1: Clear PyTorch Cache
                self.model.zero_grad()
                # 2: Convert BOW to vectors:
                bow_vec = self.make_bow_vector(instance, self.word_to_ix)
                target = self.make_target(label, self.label_to_ix)

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
                train_n += 1
            
            train_ellapse = time.time() - train_start
    
            # TEST -----------------------------:
            i, n = 0, len(self.pytorch_data_test)
            with torch.no_grad(): # Not training!
                for instance, label in self.pytorch_data_test:
                    i += 1
                    print(" > Validating [{}/{}]".format(i, n),  end='\r')
                    bow_vec = self.make_bow_vector(instance, self.word_to_ix)
                    target = self.make_target(label, self.label_to_ix)
                    if self.config.USE_GPU:
                        bow_vec = bow_vec.to(self.device)
                        target = target.to(self.device)
                    log_probs = self.model(bow_vec)
                    # Log summay:
                    loss = loss_(log_probs, target)
                    test_loss_sum += loss.item()
                    test_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
                    test_n += 1
        
            test_ellapse = time.time() - test_start
    
            # Log ------:
            report_.append(
                epoch         = epoch,
                train_loss    = train_loss_sum / train_n,
                train_acc     = train_acc_sum / train_n,
                train_time    = train_ellapse,
                test_loss     = test_loss_sum / test_n,
                test_acc      = test_acc_sum / test_n,
                test_time     = test_ellapse,
                learning_rate = 0,
                verbose       = True
            )

            if ((test_acc_sum / test_n) >= 0.55): # early stopping
                break;
        

        # OUTPUT REPORT:
        if gen_plot:
            report_.output_progress_plot(
                figsize       = (15,12),
                OUT_DIR       = self.config.OUTPUT_FOLDER,
                tag           = self.config.MODEL_TAG
            )
        return report_
    
    def predict(self, pd_data, tag=None):
        # pre-process:
        self._print("Pre-processing Test Dataset ...")
        pd_data_processed = self.generate_tweet_message_normalized_column(
            pd_data = pd_data, config = self.config
        )
        path = abspath('processed_data/test-[{}-{}].csv'.format(self.config.MODEL_TAG, tag))
        self._print("Processed Test Dataset Saved @ {}".format(path))
        self.training_dataset.to_csv(path)
        # prediction:
        self._print("Predicting ...")
        y_pred = []
        with torch.no_grad(): # Not training!
            for x in TEST_DATA_X[self.config.PROCESSED_TWEET_TAG]:
                bow_vec = self.make_bow_vector(x, self.word_to_ix)
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


# %% USER PARAMS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# Pre-evaluation:
CONFIG = TwitterLikePredictor.PredictorConfiguration(
    Y_TAG                 = "likes_count",
    USE_GPU               = True,
    PROCESSED_TWEET_TAG   = "norm-tweet",
    OUTPUT_FOLDER         = abspath("output"),
    MODEL_TAG             = "test-epoch-10-v3",
    # tweet pre-processing:
    DELIMITER_SET         = '; |, |、|。| ',
    SYMBOLE_REMOVE_LIST   = [],#["\[", "\]", "\(", "\)", "#"],
    KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
    # training set:
    SHUFFLE_TRAINING      = False,
    PERCENT_TRAINING_SET  = 0.90, # 0.99
    # bag of words (Tweeter Interpretation):
    LOSS_FUNC             = nn.NLLLoss(),
    BOW_TOTAL_NUM_EPOCHS  = 10, # 20
    LEARNING_RATE         = 0.001,
    # PERCENT_TRAINING_SET  = 0.9,
    # # Best Model so far: test-epoch-1 (Incorporate time ... info.) => 0.462 acc --> 0.45604
    # BOW_TOTAL_NUM_EPOCHS  = 10,
    # LEARNING_RATE         = 0.001,
    # # Best Model so far: test-epoch-1 => 0.4365 acc --> 0.44792
    # BOW_TOTAL_NUM_EPOCHS  = 20,
    # LEARNING_RATE         = 0.001,
    # # Best Model so far: test-epoch-180
    # BOW_TOTAL_NUM_EPOCHS  = 180,
    # LEARNING_RATE         = 0.001,
)

# ## INIT: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
TLP_Engine = TwitterLikePredictor(pd_data_training=TRAIN_DATA, verbose=True, config=CONFIG)

# %% TODO: Feature reductions
list_top100 = list(map(np.array, zip(* TLP_Engine.word_count_top_100)))
# Plot Language and video Count:
fig = plt.figure(figsize=(40,40))
ax = plt.subplot(1, 1, 1)
ax.set_title("Top 100 Repeated Word Count")
plt.bar(list_top100[0], list_top100[1][:, 0], label="0")
plt.bar(list_top100[0], list_top100[1][:, 1], label="1")
plt.bar(list_top100[0], list_top100[1][:, 2], label="2")
plt.bar(list_top100[0], list_top100[1][:, 3], label="3")

# %% TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
report = TLP_Engine.train(gen_plot=True)

# %% PREDICTION: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
pd_data_processed, df_pred = TLP_Engine.predict(pd_data=TEST_DATA_X, tag="test")


# %% Save model:
torch.save(TLP_Engine.model.state_dict(), abspath("output/model-{}.pt".format(TLP_Engine.config.MODEL_TAG)))

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
n = len(TLP_Engine.pytorch_data_test)
validation_data["pred-likes"] = [0] * n
validation_data["pred-ifcorrect"] = [False] * n
validation_data["pred-probabilities"] = [[0,0,0,0]] * n
loss_ = TLP_Engine.config.LOSS_FUNC
with torch.no_grad(): # Not training!
    i = -1
    for instance, label in TLP_Engine.pytorch_data_test:
        i += 1
        print(" > Predicting [{}/{}]".format(i+1, n),  end='\r')
        bow_vec = TLP_Engine.make_bow_vector(instance, TLP_Engine.word_to_ix)
        target = TLP_Engine.make_target(label, TLP_Engine.label_to_ix)
        if TLP_Engine.config.USE_GPU:
            bow_vec = bow_vec.to(TLP_Engine.device)
            target = target.to(TLP_Engine.device)
        log_probs = TLP_Engine.model(bow_vec)
        y_pred = log_probs.argmax(dim=1).tolist()[0]
        # Log summay:
        loss = loss_(log_probs, target)

        # find index and store:
        # id_ = dict_tweet_to_id[''.join(instance)]
        id_ = TLP_Engine.pytorch_data_test_id[i]
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