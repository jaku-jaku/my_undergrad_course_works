# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from typing import Dict, Any, List
import re
import time

from sklearn.model_selection import train_test_split

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

# %% USER PARAMS: ----
# Pre-evaluation:
FIG_SIZE = (8,8)
OUTPUT_FOLDER = abspath("output")

# Pre-Processing Configurations:
DELIMITER_SET = '; |, |、|。| \) | \( | \[ | \] | '
KEYS_TO_REMOVE_LIST = ["http", "arXiv", "https"]

# Training set:
MODE_TAG = "tweet-only-WoB"
PERCENT_TRAINING_SET = 0.9
SHUFFLE_TRAINING_SET = False
LEARNING_RATE = 0.1
TOTAL_NUM_EPOCHS = 10
USE_GPU = False # TODO: to be tested!


# %% INIT: ----
# gen folder:
jx_lib.create_folder(OUTPUT_FOLDER)

# import data
TRAIN_DATA_X = pd.read_csv(abspath("data/p_train_x.csv"))
TRAIN_DATA_Y = pd.read_csv(abspath("data/p_train_y.csv"))

ic(np.sum(TRAIN_DATA_X["id"] == TRAIN_DATA_Y["id"])) # so the assumption should be right, they share the exact same id in sequence
TRAIN_DATA = pd.concat([TRAIN_DATA_X, TRAIN_DATA_Y["likes_count"]], axis=1)

ic(TRAIN_DATA.shape)


# %%[markdown]
# ## Pre-Data Analysis
# %%
TRAIN_DATA.head(10)
sns.displot(TRAIN_DATA["likes_count"])
HEADERS = list(TRAIN_DATA.columns)
a = ic(HEADERS)

# Plot Language and video Count:
fig = plt.figure(figsize=(20,20))
ax = plt.subplot(2, 2, 1)
ax.set_title("Language Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="language", hue="likes_count", multiple="dodge")

ax = plt.subplot(2, 2, 2)
ax.set_title("Video Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="video", hue="likes_count", multiple="dodge")
# %%[markdown]
# ## Translate Training Data

# %% ----------------------------------------------------------------
def generate_tweet_message_normalized_column(
        pd_data, 
        MAX_LENGTH = None, 
        DELIMITER: str = DELIMITER_SET,
        KEYS_TO_REMOVE: List[str] = KEYS_TO_REMOVE_LIST
    ):
    if MAX_LENGTH is None:
        MAX_LENGTH, d = pd_data.shape
    tweet_data = []
    for i in range(MAX_LENGTH):
        messages = pd_data['tweet'][i]
        # separate delimiter:
        messages = re.split(DELIMITER, messages)
        # remove keys:
        new_messages = []
        for msg in messages:
            no_key = True
            for key in KEYS_TO_REMOVE: # tags to be removed
                if key in msg:
                    no_key = False
            if no_key and len(msg) > 0:
                # split:
                new_messages.extend(jieba.lcut(msg, cut_all=True)) # split east asian
        tweet_data.append(new_messages)

    pd_data['norm-tweet'] = tweet_data

generate_tweet_message_normalized_column(
    pd_data = TRAIN_DATA
)

# save frame
TRAIN_DATA.to_csv(abspath('processed_data/train.csv'))

# samole:
UNIQ_LANG = TRAIN_DATA["language"].unique().tolist()
for lang in UNIQ_LANG:
    index = TRAIN_DATA.index[TRAIN_DATA["language"] == lang].tolist()[0]
    print(lang, " >", TRAIN_DATA["norm-tweet"][index])

# %% ---- ---- ---- ---- ---- NLP - Bag of words:
# gen pytorch data:
def pandas2pytorch(
    pd_data,
    x_tag,
    y_tag,
    range
):
    return [(msg, like) for msg, like in zip(pd_data[x_tag][range[0]:range[1]], pd_data[y_tag][range[0]:range[1]])]

# let's shuffle the training data:
if SHUFFLE_TRAINING_SET:
    TRAIN_DATA = TRAIN_DATA.sample(frac = 1)
N_TRAIN = int(len(TRAIN_DATA) * PERCENT_TRAINING_SET)
N_TEST = len(TRAIN_DATA) - N_TRAIN

# let's split data
pytorch_data_train = pandas2pytorch(
    pd_data = TRAIN_DATA,
    x_tag = "norm-tweet", y_tag = "likes_count",
    range =[0, N_TRAIN]
)
pytorch_data_test = pandas2pytorch(
    pd_data = TRAIN_DATA,
    x_tag = "norm-tweet", y_tag = "likes_count",
    range =[N_TRAIN, N_TRAIN+N_TEST]
)

ic(np.shape(pytorch_data_train))
ic(np.shape(pytorch_data_test))


# %% generate a word to index:
word_to_ix = {}
word_count = {}
for sent, _ in pytorch_data_train + pytorch_data_test:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            word_count[word] = 0
        else:
            word_count[word] += 1
            
VOCAB_SIZE = len(word_to_ix)
ic(VOCAB_SIZE)

# sample top bag of words
word_count_top_100 = sorted(word_count.items(), key=lambda x:-x[1])[:100]
ic(word_count_top_100)

# %% Train:
# LOAD NET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# check device:
# hardware-acceleration
device = None
if torch.cuda.is_available():
    print("[ALERT] Attempt to use GPU => CUDA:0")
    device = torch.device("cuda:0")
else:
    print("[ALERT] GPU not found, use CPU!")
    device =  torch.device("cpu")
# MODEL_DICT["VGG11"].to(device)

#####
data = pytorch_data_train
test_data = pytorch_data_test

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 4

class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        # do not use word if it was not in the dictionary, this happens when unseen testing dataset
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    ic(param)

# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    ic(bow_vector)
    ic(log_probs)


label_to_ix = {0: 0, 1: 1, 2:2, 3:3}

make_target(3, label_to_ix)
# %% ---- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# ------- ------- ------- TRAIN:
report = ProgressReport()
if USE_GPU:
    model.to(device)
    
# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(TOTAL_NUM_EPOCHS):
    print("> epoch {}/{}:".format(epoch + 1, TOTAL_NUM_EPOCHS))
    
    train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
    test_loss_sum, test_acc_sum, test_n, test_start = 0.0, 0.0, 0, time.time()

    # TRAIN -----------------------------:
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        if USE_GPU:
            bow_vec = bow_vec.to(device)
            target = target.to(device)
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        # Log summay:
        train_loss_sum += loss.item()
        train_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
        train_n += 1
    
    train_ellapse = time.time() - train_start
    
    # TEST -----------------------------:
    # with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        if USE_GPU:
            bow_vec = bow_vec.to(device)
            target = target.to(device)
        log_probs = model(bow_vec)
        # Log summay:
        test_loss_sum += loss.item()
        test_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
        test_n += 1
    
    test_ellapse = time.time() - test_start
    
    # Store ------:
    report.append(
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

# OUTPUT REPORT:
report.output_progress_plot(
    figsize       = (15,12),
    OUT_DIR       = OUTPUT_FOLDER,
    tag           = MODE_TAG,
)
# %%


# %%
TEST_DATA_X = pd.read_csv(abspath("data/p_test_x.csv"))
generate_tweet_message_normalized_column(
    pd_data = TEST_DATA_X
)
TEST_DATA_X.to_csv(abspath('processed_data/test_x.csv'))
y_pred = []
for x in TEST_DATA_X['norm-tweet']:
    bow_vec = make_bow_vector(x, word_to_ix)
    if USE_GPU:
        bow_vec = bow_vec.to(device)
    log_probs = model(bow_vec)
    y_pred.append(log_probs.argmax(dim=1))


# %%
df_pred = pd.DataFrame({'label':[y.tolist()[0] for y in y_pred]})
df_pred.to_csv(abspath('processed_data/test_y_pred_[{}].csv'.format(MODE_TAG)),index_label="id")

# %%
