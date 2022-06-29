# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Activation
from tensorflow.keras.layers import Flatten, Embedding, Input, GRU, Bidirectional
from tensorflow.keras.layers import LSTM, SimpleRNNCell, RNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
import spacy
from sklearn.metrics import f1_score, roc_auc_score
import project_functions as pf


# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# GENERAL

def pred_labels(model, X_test):

    """
    Function to make predictions for given X data and instead of
    returning raw probabilities, returns the predictions in label
    form: 0 = true news, 1 = fake news.
    ----------------------------------------------------------------------
    model =  a trained model to predict with
    X_test = array of features you want to predict from
    """

    # Make predictions / get raw probabilities
    raw_preds = model.predict(X_test)

    # Convert probabilities into labels
    preds = (raw_preds > 0.5).astype(int).reshape(-1)

    return preds


def pred_text_as_labels(text, tokenizer,
    model, maxlen = 500,
     truncating = "post",
     as_labels = True):
    """
    Function to make predictions for new texts/articles using a fitted tokenizer
    and a fitted model. Returns either label predictions (0/1) or probabilities
    for whether the articles are true/fake, depending on as_labels arg.
    ----------------------------------------------------------------------
    text = list of strings, texts/articles you want to try and classify
    tokenizer = a fitted Keras Tokenizer() object
    model =  a trained model to predict with
    max_len = int, maximum length of sequences
    padding = str, whether sequences should be padded at the front or back
    truncating = str, whether sequences should be truncated at the front or back
    as_labels = bool, if True will return model predictions in label form- 0 = true and
    1 = fake. If False, gives the raw probabilities.
    """

    # Convert text to sequences
    seqs = tokenizer.texts_to_sequences(text)
    # Pad/trim sequences
    padded_seqs = pad_sequences(seqs, maxlen = maxlen, truncating = truncating)
    # Make predictions
    raw_preds = model.predict(padded_seqs)

    if as_labels:
        # Convert from probabilities into labels
        preds = (raw_preds > 0.5).astype(int).reshape(-1)
        return preds

    else:
        return raw_preds


def get_test_metrics(model, X_test, y_test, all_results,
                     history,
                    name = "mlp",
                    embedding = None,
                    regularize = False,
                    batch_normalize = False,
                    verbose = 0):
    """
    Evaluates a fitted model against test data in terms of accuracy,
    ROC AUC score and F1 score. Must be called after the embedding layer,
    regularize toggle and batch_normalize toggle have all been instantiated.
    Returns a df of model name, specs and test metrics.
    ----------------------------------------------------------------------
    model =  a trained model to predict with
    X_test = array of padded text sequences
    y_test = int, 0/1 labels for true/fake news
    all_results = pd.DataFrame, either empty or containing rows of
    other models' results
    history = Keras history object
    name = str, the name of the architecture being used
    embedding = str/None, name of embedding layer used in embedding dict
    regularize = bool, whether or not model was regularized
    batch_normalize = bool, whether or not model was batch normalized
    verbose = int, controls messaging of model.evaluate
    """
    
    # Get number of epochs run for
    n_epochs = len(history.history["loss"])

    # Get test accuracy
    test_acc = model.evaluate(X_test, y_test, verbose = verbose)[1]

    # Get raw predictions / probabilities for ROC AUC
    probs = model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, probs)

    # Get label predictions for F1
    preds = pred_labels(model, X_test)
    test_f1 = f1_score(y_test, preds)

    # Save results
    results = pd.DataFrame({"name":name,
                            "embedding":embedding,
                            "regularize":regularize,
                            "batch_normalize":batch_normalize,
                            "accuracy":test_acc,
                           "roc_auc":test_roc_auc, "f1":test_f1,
                           "epochs":n_epochs}, index = [0])

    # Store results with others
    all_results = pd.concat([all_results, results])
    return all_results


def fit_and_save(X_train, y_train, model,
                 name = "mlp",
                 embedding = None,
                 regularize = False,
                 batch_normalize = False,
                 save_model = False,
                 save_history = True,
                 **kwargs):
    """
    Fits a Keras model and if desired, save both the model and its history
    into folders following the naming convention models/name/
    embedding_{additional_layers}. Returns a Keras history object.
    Designed to work in analysis NB when regularise and batch_normalize
    have been instantiated.
    ----------------------------------------------------------------------
    X_train = array of padded text sequences
    y_train = int, 0/1 labels for true/fake news
    model =  a trained model to predict with
    name = str, the name of the architecture being used
    embedding_layer = str/None, name of embedding layer used in embedding dict
    regularize = bool, whether or not model was regularized
    batch_normalize = bool, whether or not model was batch normalized
    save = bool, whether or not to save history and model
    **kwargs = fit_hp dict of args that Keras model.fit can take
    """

    history = model.fit(X_train, y_train, **kwargs)

    # Make filepath
    save_path = str(embedding)
    if regularize:
        save_path += "_reg"
    if batch_normalize:
        save_path += "_bn"

    # Save model if desired
    if save_model:
        # Create folder if it doesn't exist
        if not os.path.exists(f"models/{name}"):
            os.makedirs(f"models/{name}")
        model.save(f"models/{name}/{save_path}")

    # Save model if desired
    if save_history:
        # Make into df
        hist_df = pd.DataFrame(history.history)
        # Create folder if it doesn't exist
        if not os.path.exists(f"histories/{name}"):
            os.makedirs(f"histories/{name}")
        hist_df.to_csv(f"histories/{name}/{save_path}.csv", index = False)

    return history


# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Embeddings


def spacy_embedding(tokenizer, maxlen = 500, show_progress = False):

    """Function to create SpaCy embedding layer. Uses the en_core_web_sm pipeline,
    a small English-language pipeline appropriate for blogs, news and comments.
    This takes a while to run.
    ----------------------------------------------------------------------
    tokenizer = a fitted Keras Tokenizer() object
    max_len = int, maximum length of sequences
    show_progress = bool, simple indicator to tell you how much progress with
    the embedding you have made as %.
    """

    # Load the spacy pipeline
    # small English pipeline trained on written web text (blogs, news, comments)
    nlp = spacy.load("en_core_web_sm")
    # Get vocab size of tokenizer
    vocab_size = len(tokenizer.word_index) + 1

    # Get the number of embedding dimensions SpaCy uses
    embedding_dim = nlp("any_word").vector.shape[0]
    # Create a matrix to use in embedding layer
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Iterate through our vocabulary, mapping words to spacy embedding
    # this will take a while to run
    for i, word in enumerate(tokenizer.word_index):
        embedding_matrix[i] = nlp(word).vector
        # Show progress if desired
        if show_progress:
            if i % 10000 == 0 and i > 0:
                print(round(i*100/vocab_size, 3), "% complete")


    # Load the embedding matrix as the weights matrix for the embedding layer
    # Set trainable to False as the layer is already "learned"
    Embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        input_length = maxlen,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
        name = "spacy_embedding")

    return Embedding_layer


def glove_embedding(tokenizer,
                    filepath = "../glove/glove.6B.300d.txt",
                    maxlen = 500,
                     show_progress = False):

    """Function to create GloVe embedding layer. Uses the Wikipedia 2014 and Gigaword 5 dataset.
    It's trained in English and features News data so is appropriate for this task.
    ----------------------------------------------------------------------
    tokenizer = a fitted Keras Tokenizer() object
    filepath = str, path to the glove pre-trained vector file
    max_len = int, maximum length of sequences
    show_progress = bool, simple indicator to tell you how much progress with
    the embedding you have made as %.
    """

    # Create dict to store glove embeddings in- word:vector
    glove_embeddings = {}

    # Load the GloVe embeddings
    # trained on combination of wikipedia and news data (English language)
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_embeddings[word] = vector

    # Get vocab size of tokenizer
    vocab_size = len(tokenizer.word_index) + 1

    # Get the number of embedding dimensions GloVe uses
    embedding_dim = vector.shape[0]
    # Create a matrix to use in embedding layer
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Iterate through our vocabulary, mapping words to GloVe embedding
    for i, word in enumerate(tokenizer.word_index):
        # Try to find corresponding vector for word, else return None
        embedding_vector = glove_embeddings.get(word)
        # If word exists, update matrix with vector
        # Words that couldn't be mapped are 0s in the matrix
        if embedding_vector is not None:
            embedding_matrix[i] = glove_embeddings[word]
        # Display progress if desired
        if show_progress:
            if (i % 10000 == 0 and i > 0):
                print(round(i*100/vocab_size, 3), "% complete")

    # Load the embedding matrix as the weights matrix for the embedding layer
    # Set trainable to False as the layer is already "learned"
    Embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        input_length = maxlen,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
        name = "glove_embedding")

    return Embedding_layer


def keras_embedding(tokenizer, embedding_dim = 256, maxlen = 500):

    """Function to create a custom Keras embedding layer.
    ----------------------------------------------------------------------
    tokenizer = a fitted Keras Tokenizer() object
    max_len = int, maximum length of sequences
    """

    # Get vocab size of tokenizer
    vocab_size = len(tokenizer.word_index) + 1

    # Load the embedding matrix as the weights matrix for the embedding layer
    # Set trainable to False as the layer is already "learned"
    Embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        input_length = maxlen,
        name = "keras_embedding")

    return Embedding_layer

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# MLP

def mlp(loss = "binary_crossentropy",
                optimizer = "adam",
               metrics = ["accuracy"], regularize = False,
               batch_normalize = False,
               embedding = None,
               maxlen = 500,
               hidden_dense_units = 256,
               dense_kernel_initializer = "he_uniform"):
    """
    Creates an MLP designed to be used with the text data only.
    Can be built with either Keras, SpaCy, GloVe or no embedding.
    Returns a compiled Keras model of 3 Dense layers (1024, 256, 1). There are
    options to include elasticnet regularization and batch normalisation layers too.
    ----------------------------------------------------------------------
    loss = str, name of loss function to use
    optimizer = Keras optimizer, set to 'adam' but any optimizer can be passed
    metrics =  list of Keras metrics to use to evaluate with
    regularize = bool, if True adds elasticnet/l1_l2 regularisation with
    l1 = 0.01 and l2 = 0.01
    batch_normalize = bool, if True adds batch normalisation between hidden Dense
    and output layer.
    embedding = None/Keras embedding instance: The type of embedding to use (SpaCy,
    GloVe, Keras or none).
    maxlen = int, shape of input (length of sequences)
    hidden_dense_units = int, number of hidden units in the hidden dense layer.
    dense_kernel_initializer = str or keras.initializers object for the weights
    of the Dense layer.
    """

    # Build model
    model = Sequential(name = "MLP")

    # Add embedding if desired
    if embedding:
        # Embedding contains input shape
        model.add(embedding)
        # Flatten embeddings
        model.add(Flatten())
    else:
        model.add(Input(shape = (maxlen, ), name = "Input"))

    # Elasticnet regularised model
    if regularize:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense_Elasti",
                        kernel_regularizer = l1_l2(),
                        kernel_initializer = dense_kernel_initializer))

    # Baseline model
    else:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense",
                        kernel_initializer = dense_kernel_initializer))

    # Batch normalised model
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_Norm1"))

    # Apply non-linear activation, specified in this way to be consistent
    # with the original paper
    model.add(Activation("relu", name = "ReLU_Activation"))

    # Output layer
    model.add(Dense(1, activation = "sigmoid", name = "Output",
                    kernel_initializer = dense_kernel_initializer))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer,
                  metrics = metrics)

    return model

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# RNN

def bi_rnn(embedding = None, maxlen = 500,
           rnn_units = 32, rnn_kernel_initializer = "glorot_uniform",
           hidden_dense_units = 256, dense_kernel_initializer = "he_uniform",
           regularize = False, batch_normalize = False,
           loss = 'binary_crossentropy', optimizer = 'adam',
           metrics = ['accuracy']):

    """
    Instantiates a bidirectional RNN model used to detect fake news. The basic
    structure includes one of three possible embeddings (Keras trainable embedding,
    GloVe, and SpaCy), plus no embedding (the default). Afterwards, there is one
    bidirectional recurrent layer that relies on rnn gates.
    After the RNN layer there is another fully-connected layer to expand
    the interactions of the RNN output before the final output layer. A ReLu
    activation function is applied here.
    Finally, since the task is binary classification, a dense layer with sigmoid
    activation function is included.

    Summary of the architecture:
    Embedding - Bidirectional RNN 32 * 2 units - Dense layer 256 units with ReLu -
        Output dense layer

    Further modifications can be included, either a kernel regularizer
    or a batch normalization layer to improve generalization. Both
    will be applied either in or right after the hidden dense layer.

    Finally, this function compiles and returns the model.
    ----------------------------------------------------------------------
    embedding = Embedding matrix object already loaded into memory
    maxlen = Maximum length of each padded sequence (interpreted as number of timesteps)
    rnn_units = int, number of hidden units in the recurrent computation of RNN
    rnn_kernel_initializer = str or keras.initializers object for the weights
        of the RNN layer. Default as specified in the keras function.
    hidden_dense_units = int, number of hidden units in the hidden dense layer
    dense_kernel_initializer = str or keras.initializers object for the weights
        of the Dense layer. Default as specified in the keras function.
    regularize = bool, application of elasticnet to the weights of the hidden
        dense layer through the use of the l1_l2() keras function,
        with l1 = 0.01 and l2 = 0.01
    batch_normalize = bool, application of Batch Normalization after the hidden
        dense layer
    loss = str or keras.losses object, specifies the loss function
    optimizer = str or keras.optimizers object
    metrics = list of str or keras.metrics objects, metrics to be calculated
    """

    # Set seed to ensure reproducibility across different notebooks (this
    # can be changed by the user)
    model = Sequential(name = "RNN")

    # a) Embeddings: Add the kind of embedding the user requires.
    #   1. Any embedding as stored in the kernel by the user (such as GloVe or Keras):
    if embedding:
        model.add(embedding)

    #   2. Default: No embedding. We specify the shape as
    #   (number of timesteps, n_features). With only text, n_features = 1
    #   Batch_size is already guessed by TF while fitting.
    else:
        model.add(Input(shape = (maxlen, 1), name = "Input"))

    # b) RNN bidirectional layer
    # First we instantiate the basic rnn_cell, which is the basic component
    # of the RNN layer. Then, we create a Bidirectional RNN layer
    rnn_cell = SimpleRNNCell(rnn_units, kernel_initializer = rnn_kernel_initializer)

    model.add(Bidirectional(RNN(rnn_cell), name = "Bidirectional_RNN"))

    # c) Densely connected layer, where regularization can be applied
    if regularize:
        model.add(Dense(hidden_dense_units,
                        kernel_initializer = dense_kernel_initializer,
                        kernel_regularizer = l1_l2(),
                        name = "Regularized_hidden_dense"))
    else:
        model.add(Dense(hidden_dense_units,
                        kernel_initializer = dense_kernel_initializer,
                        name = "Hidden_dense"))

    # d) Batch normalization layer, added if specified:
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_normalization"))

    # Activation function of the dense layer, applied after batch normalization
    # in case this has been specified.
    model.add(Activation("relu", name = "ReLu_activation"))

    # e) Final Dense layer:
    model.add(Dense(1, activation = "sigmoid",
                    kernel_initializer = dense_kernel_initializer,
                    name = "Output"))


    # Compile the model with the user specifications:
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# LSTM

def bi_lstm(embedding = None, maxlen = 500,
            rnn_units = 32, rnn_kernel_initializer = "glorot_uniform",
            hidden_dense_units = 256, dense_kernel_initializer = "he_uniform",
            regularize = False, batch_normalize = False,
            loss = 'binary_crossentropy', optimizer = 'adam',
            metrics = ['accuracy']):

    """
    Instantiates a bidirectional LSTM model used to detect fake news. The basic
    structure includes one of three possible embeddings (Keras trainable embedding,
    GloVe, and SpaCy), plus no embedding (the default). Afterwards, there is one
    bidirectional recurrent layer that relies on LSTM gates.
    After the LSTM layer there is another fully-connected layer to expand
    the interactions of the LSTM output before the final output layer. A ReLu
    activation function is applied here.
    Finally, since the task is binary classification, a dense layer with sigmoid
    activation function is included.

    Summary of the architecture:
    Embedding - Bidirectional LSTM 32 * 2 units - Dense layer 256 units with ReLu -
        Output dense layer

    Further modifications can be included, either a kernel regularizer
    or a batch normalization layer to improve generalization. Both
    will be applied either in or right after the hidden dense layer.

    Finally, this function compiles and returns the model.
    ----------------------------------------------------------------------
    embedding = Embedding matrix object already loaded into memory
    maxlen = Maximum length of each padded sequence (interpreted as number of timesteps)
    rnn_units = int, number of hidden units in the recurrent computation of LSTM
    rnn_kernel_initializer = str or keras.initializers object for the weights
        of the LSTM layer. Default as specified in the keras function.
    hidden_dense_units = int, number of hidden units in the hidden dense layer
    dense_kernel_initializer = str or keras.initializers object for the weights
        of the Dense layer. Default as specified in the keras function.
    regularize = bool, application of elasticnet to the weights of the hidden
        dense layer through the use of the l1_l2() keras function,
        with l1 = 0.01 and l2 = 0.01
    batch_normalize = bool, application of Batch Normalization after the hidden
        dense layer
    loss = str or keras.losses object, specifies the loss function
    optimizer = str or keras.optimizers object
    metrics = list of str or keras.metrics objects, metrics to be calculated
    """

    # Set seed to ensure reproducibility across different notebooks (this
    # can be changed by the user)
    model = Sequential(name = "LSTM")

    # a) Embeddings: Add the kind of embedding the user requires.
    #   1. Any embedding as stored in the kernel by the user (such as GloVe or Keras):
    if embedding:
        model.add(embedding)

    #   2. Default: No embedding. We specify the shape as
    #   (number of timesteps, n_features). With only text, n_features = 1
    #   Batch_size is already guessed by TF while fitting.
    else:
        model.add(Input(shape = (maxlen, 1), name = "Input"))

    # b) LSTM bidirectional layer
    model.add(Bidirectional(LSTM(rnn_units,
                                 kernel_initializer = rnn_kernel_initializer),
                                 name = "Bidirectional_LSTM"))

    # c) Densely connected layer, where regularization can be applied
    if regularize:
        model.add(Dense(hidden_dense_units,
                        kernel_initializer = dense_kernel_initializer,
                        kernel_regularizer = l1_l2(),
                        name = "Regularized_hidden_dense"))
    else:
        model.add(Dense(hidden_dense_units,
                        kernel_initializer = dense_kernel_initializer,
                        name = "Hidden_dense"))

    # d) Batch normalization layer, added if specified:
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_normalization"))

    # Activation function of the dense layer, applied after batch normalization
    # in case this has been specified.
    model.add(Activation("relu", name = "ReLu_activation"))

    # e) Final Dense layer:
    model.add(Dense(1, activation = "sigmoid",
                    kernel_initializer = dense_kernel_initializer,
                    name = "Output"))


    # Compile the model with the user specifications:
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model



# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# GRU

def bi_gru(loss = "binary_crossentropy",
                optimizer = "adam",
               metrics = ["accuracy"], regularize = False,
               batch_normalize = False,
               embedding = None,
               maxlen = 500,
               hidden_dense_units = 256,
               dense_kernel_initializer = "glorot_uniform",
               rnn_units = 32,
               rnn_kernel_initializer = "glorot_uniform"):
    """
    Creates a GRU model designed to be used with the text data only.
    Can be built with either Keras, SpaCy, GloVe or no embedding.
    Returns a compiled Keras model of a bidirectional GRU (32*2) and 2 Dense layers (256, 1).
    There are options to include elasticnet regularization and batch normalisation layers too.
    ----------------------------------------------------------------------
    loss = str, name of loss function to use
    optimizer = Keras optimizer, set to 'adam' but any optimizer can be passed
    metrics =  list of Keras metrics to use to evaluate with
    regularize = bool, if True adds elasticnet/l1_l2 regularisation with
    l1 = 0.01 and l2 = 0.01
    batch_normalize = bool, if True adds batch normalisation between hidden Dense
    and output layer.
    embedding = None/Keras embedding instance: The type of embedding to use (SpaCy,
    GloVe, Keras or none).
    maxlen = int, shape of input (length of sequences)
    hidden_dense_units = int, number of hidden units in the hidden dense layer
    dense_kernel_initializer = str or keras.initializers object for the weights
    of the Dense layer.
    rnn_units = int, number of hidden units in the recurrent computation of GRU.
    rnn_kernel_initializer = str or keras.initializers object for the weights
    of the GRU layer.
    """

    # Build model
    model = Sequential(name = "GRU")

    # Add embedding if desired
    if embedding:
        # Embedding contains input shape
        model.add(embedding)
    else:
        # Otherwise reshape data to work with GRU
        model.add(Reshape((maxlen, 1), input_shape = (maxlen, ), name = "Reshaping"))

    # Add GRU
    model.add(Bidirectional(GRU(rnn_units,
                                kernel_initializer = rnn_kernel_initializer),
                                name = "Bidirectional_GRU"))

     # Elasticnet regularised model
    if regularize:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense_Elasti",
                         kernel_regularizer = l1_l2(),
                          kernel_initializer = dense_kernel_initializer))

    # Baseline model
    else:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense",
                        kernel_initializer = dense_kernel_initializer))

    # Batch normalised model
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_Norm1"))

    # Apply non-linear activation, specified in this way to be consistent
    # with the original paper
    model.add(Activation("relu", name = "ReLU_Activation"))


    # Output layer
    model.add(Dense(1, activation = "sigmoid", name = "Output",
                    kernel_initializer = dense_kernel_initializer))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer,
                  metrics = metrics)


    return model

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# CNN

def cnn(loss = "binary_crossentropy",
                optimizer = "adam",
               metrics = ["accuracy"], regularize = False,
               batch_normalize = False,
               embedding = None,
               maxlen = 500,
               conv_filters = 256,
               conv_kernel_size = 8,
               pool_size = 2,
               hidden_dense_units = 256,
               dense_kernel_initializer = "he_uniform"):
    """
    Creates a Convolutional Neural Network designed to be used with the text data only.
    Can be built with either Keras, SpaCy, GloVe or no embedding.
    Returns a compiled Keras model of 1 Conv1D layer, 1 MaxPooling1D layer, and 2 Dense layers (256, 1).
    There are options to include elasticnet regularization and batch normalisation layers too.
    ----------------------------------------------------------------------
    loss = str, name of loss function to use
    optimizer = Keras optimizer, set to 'adam' but any optimizer can be passed
    metrics =  list of Keras metrics to use to evaluate with
    regularize = bool, if True adds elasticnet/l1_l2 regularisation with
    l1 = 0.01 and l2 = 0.01
    batch_normalize = bool, if True adds batch normalisation between hidden Dense
    and output layer.
    embedding = None/Keras embedding instance: The type of embedding to use (SpaCy,
    GloVe, Keras or none).
    maxlen = int, shape of input (length of sequences).
    conv_filters = int, the dimensionality of the Convolutional layer output space.
    conv_kernel_size = int, specifying the length of the 1D convolution window.
    pool_size = int, size of the max pooling window.
    hidden_dense_units = int, number of hidden units in the hidden dense layer.
    dense_kernel_initializer = str or keras.initializers object for the weights
    of the Dense layer.
    """

    # Build model
    model = Sequential(name = "CNN")

    # Add embedding if desired
    if embedding:
        # Embedding contains input shape
        model.add(embedding)
    else:
        model.add(Reshape((maxlen, 1), input_shape = (maxlen, ), name = "Reshaping"))

    #Add Conv1D Layer
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', name = "Conv_layer"))

    #Add MaxPooling1D Layer
    model.add(MaxPooling1D(pool_size=pool_size))

    #Flatten embeddings before passing to Dense layers
    model.add(Flatten())

    # Elasticnet regularised model
    if regularize:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense_Elasti",
                        kernel_regularizer = l1_l2(),
                        kernel_initializer = dense_kernel_initializer))

    # Baseline model
    else:
        model.add(Dense(hidden_dense_units, name = "Linear_Dense",
                        kernel_initializer = dense_kernel_initializer))

    # Batch normalised model
    if batch_normalize:
        model.add(BatchNormalization(name = "Batch_Norm1"))

    # Apply non-linear activation, specified in this way to be consistent
    # with the original paper
    model.add(Activation("relu", name = "ReLU_Activation"))

    # Output layer
    model.add(Dense(1, activation = "sigmoid", name = "Output",
                    kernel_initializer = dense_kernel_initializer))
    # Compile model
    model.compile(loss = loss, optimizer = optimizer,
                  metrics = metrics)

    return model
