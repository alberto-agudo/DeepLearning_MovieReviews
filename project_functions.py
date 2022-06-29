# IMPORTS
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.random import set_seed as tf_set_seed
import random

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# PRE-PROCESSING

def imdb_read_dir(imdb_dir):
    """
    Reads an IMDB directory given a pathfile (imdb_dir) and returns
    a DataFrame with a column for the review and another one for the sentiment.
    """
    reviews = []
    
    # Get sentiment from filename
    sentiment = imdb_dir.split("/")[-2]
    
    # Map sentiments to values
    sent_dict = {"pos":1, "neg":0}

    # Read in files
    for file in os.listdir(imdb_dir):
        with open(imdb_dir + file, "r") as f:
            reviews.append(f.readlines())
    
    # Load into dataframe
    df = pd.DataFrame(reviews, columns = ["review"])
    df["sentiment"] = sent_dict[sentiment]

    return df



def imdb_train_test_dfs(train_path_neg = "../aclImdb/train/neg/",
                        train_path_pos = "../aclImdb/train/pos/",
                        test_path_neg = "../aclImdb/test/neg/",
                        test_path_pos = "../aclImdb/test/pos/",
                       original_test_train_split = False):
    """
    Gets the path of training and test data for both positives and negatives
    and returns a train and a test dataframe. They give the data as a 50:50 
    test train split, we instead go for a 80:20 split later on so combine
    the data.
    """
    # Get train data
    train_neg = imdb_read_dir(train_path_neg)
    train_pos = imdb_read_dir(train_path_pos)
    train_df = pd.concat([train_neg, train_pos], axis = 0).reset_index(drop = True)

    # Get test data
    test_neg = imdb_read_dir(test_path_neg)
    test_pos = imdb_read_dir(test_path_pos)
    test_df = pd.concat([test_neg, test_pos], axis = 0).reset_index(drop = True)
    
    # If the original split of data is desired
    if original_test_train_split:
        return train_df, test_df
    
    # Else combine for 80:20 splitting later
    else:
        return pd.concat([train_df, test_df]).reset_index(drop = True)
    



def tokenize_padder(train_text, test_text,
                   chars_to_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                    oov_token = "OOV",
                    maxlen = 500,
                    padding = "pre",
                    truncating = "post"
                   ):
    """
    Function to train a Keras Tokenizer on training text data, then use that to
    generate sequences for both training and text data. It then pre-pads those
    sequences that are too short and post-trims any that are too long, ensuring
    they are all the same length. Returns fitted Tokenizer object,
    padded training data and padded test data.
    Mean/median token counts in the dataset were around 400so a maxlen of 500
    may be sufficient.
    ----------------------------------------------------------------------
    chars_to_filter = str, regex pattern of chars to be removed by the
    tokenizer
    oov_token = str, string representation for out-of-vocabulary tokens
    max_len = int, maximum length of sequences
    padding = str, whether sequences should be padded at the front or back
    truncating = str, whether sequences should be truncated at the front or back
    """
    # Create tokenizer
    tokenizer = Tokenizer(filters = chars_to_filter,
                          oov_token = oov_token)

    # Fit tokenizer on training data only
    tokenizer.fit_on_texts(train_text)

    # Generate sequences
    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)

    # Pad and trim sequences
    # Pre-padding is empirically better for sequence modelling
    # Post-truncating ensures the titles are included in observations
    train_padded = pad_sequences(train_sequences, maxlen = maxlen, padding = padding, truncating = truncating)
    test_padded = pad_sequences(test_sequences, maxlen = maxlen, padding = padding, truncating = truncating)

    return tokenizer, train_padded, test_padded

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# PLOTTING + FORMATTING

def set_plot_config():
    """"Function to set-up Matplotlib plotting config
    for neater graphs"""
    plt.rcParams["figure.figsize"] = (17, 8)
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    plt.rc('font', **font)


# Class to make print statements prettier
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_loss(history, xlabel = "Epochs", ylabel = "Loss",
             title = "Loss vs Epochs", ylim_top = None,
             metric = "loss", metric2 = "val_loss",
             metric_label = "Training Loss", metric2_label = "Validation Loss"):
    """Function to plot model loss.
    Takes history, a model history instance.
    Displays plot inline"""
    # Show training and validation loss vs epochs
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label = metric_label, lw = 3)
    # Add in control if you only want to plot one metric
    if metric2:
        ax.plot(history.history[metric2], label = metric2_label, lw = 3)
        ax.legend()
    # Add titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top = ylim_top)



def plot_acc(history, xlabel = "Epochs", ylabel = "Accuracy",
             title = "Accuracy vs Epochs", ylim_top = None,
             metric = "accuracy", metric2 = "val_accuracy",
             metric_label = "Training Accuracy", metric2_label = "Validation Accuracy"):
    """Function to plot model accuracy.
    Takes history, a model history instance.
    Displays plot inline"""
    # Show training and validation loss vs epochs
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label = metric_label, lw = 3, color = "lightgreen")
    # Add in control if you only want to plot one metric
    if metric2:
        ax.plot(history.history[metric2], label = metric2_label, lw = 3, color = "green")
        ax.legend()
    # Add titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top = ylim_top)


def plot_and_save(history, save = True, 
                  name = "mlp",
                  embedding = None, 
                  regularize = False,
                  batch_normalize = False,
                  display = False):
    """
    Takes a Keras/model history and plots its validation vs normal accuracies
    and losses. The displaying and saving of these plots can be toggled.
    Designed to work in analysis NB when regularise and batch_normalize
    have been instantiated.
    ----------------------------------------------------------------------
    history = Keras history object
    save = bool, whether or not to save graphs
    name = str, the name of the architecture being used
    embedding_layer = str/None, name of embedding layer used in embedding dict
    regularize = bool, whether or not model was regularized
    batch_normalize = bool, whether or not model was batch normalized
    display = bool, whether or not to display graphs
    """
    
    # Make filepath
    save_path = str(embedding)
    if regularize:
        save_path += "_reg"
    if batch_normalize:
        save_path += "_bn"
    
    plot_acc(history)
    # Save accuracy if desired
    if save:
        # Create folder if it doesn't exist
        if not os.path.exists(f"graphs/{name}/accuracy"):
            os.makedirs(f"graphs/{name}/accuracy")
        plt.savefig(f"graphs/{name}/accuracy/{save_path}.png")

    # Display/hide fig
    if not display:
        plt.close()
    
    plot_loss(history)
    # Save loss if desired
    if save:
        # Create folder if it doesn't exist
        if not os.path.exists(f"graphs/{name}/loss"):
            os.makedirs(f"graphs/{name}/loss")
        plt.savefig(f"graphs/{name}/loss/{save_path}.png")
    
    # Display/hide fig
    if not display:
        plt.close()

        

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Misc

def set_seed(my_seed=1):
    """"Function to quickly set Numpy, Tensorflow and random seeds
    so that models are reproducible."""
    # Set numpy seed
    seed(my_seed)
    # Set tensorflow seed
    tf_set_seed(my_seed)
    # Set random seed
    random.seed(my_seed)


def tf_message_toggle(log_level = "2"):
    """"Function to control the amount of messages that TF
    displays. e.g. Use it to prevent the setting GPU to xyz core
    info messages that TensorFlow displays. Useful for neatening up
    notebooks.
    ----------------------------------------------------------------------
    log_level = "0": all messages are logged (default behavior)
    log_level = "1": INFO messages are not printed
    log_level = "2": INFO and WARNING messages are not printed
    log_level = "3": INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
