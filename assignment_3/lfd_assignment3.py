#!/usr/bin/env python

"""
This script allows users to select a machine learning algorithm, relevant
hyperparameters and features to train and evaluate the model. Various arguments
can be supplied by the user to select data or to show a confusion matrix.

This script is intended to run through Google Colab and assumes that it is
stored along with the used data at /content/gdrive/MyDrive/AS3.
In addition the requirements specified in requirements.txt have to be installed
in the notebook.
"""

import argparse
import json
import logging
import random as python_random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Bidirectional
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.initializers import Constant

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# Setup logging configuration
logging.basicConfig(filename='/content/gdrive/MyDrive/AS3/results.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# Custom function to log and print so we don't have to write all parameters
# down each run.
def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default="train.txt", type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default="dev.txt",
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-bi", "--bidirectional", action="store_true",
                        help="If added, use a bidirectional LSTM")
    parser.add_argument("-e", "--embeddings", default="glove_reviews.json", type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                        help="Set the learning rate")
    parser.add_argument("-l", "--loss_function", default="categorical_crossentropy", type=str,
                        help="Set the loss function")
    parser.add_argument("-a", "--activation", default="softmax", type=str,
                        help="Set the activation")
    parser.add_argument("-ah", "--activation_hidden", default="sigmoid", type=str,
                        help="Set the activation for the hidden layer")
    parser.add_argument("-bs", "--batch_size", default=16, type=int,
                        help="Set the batch size")
    parser.add_argument("-ep", "--epochs", default=50, type=int,
                        help="Set the number of epochs")
    parser.add_argument("-m", "--momentum", default=0.9, type=float,
                        help="Controls the influence of previous epoch on the next weight update")
    parser.add_argument("-es", "--early_stop", default=3, type=int,
                        help="Set the patience of early stop")
    parser.add_argument("-o", "--optimizer", default="sgd", choices=["sgd", "adam"],
                        help="Select optimizer (SGD, ADAM)")
    parser.add_argument("-dr", "--dropout", default=None, type=float,
                        help="Set a dropout layer")
    parser.add_argument("-ex", "--extra_layer", default=0, type=int, choices=[1, 2],
                        help="Set an amount of extra layers, max 2 extra layers, keeps the same settings as the base layer.")

    parser.add_argument("-tr", "--transformer", default=None, choices=["distilbert", "roberta", "electra", "deberta"])

    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    """Read in review data set and returns docs and labels"""
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    embeddings = json.load(open(embeddings_file, "r"))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix, args):
    """Create the Keras model to use"""

    # Define hyperparameter settings
    learning_rate = args.learning_rate
    loss_function = args.loss_function
    momentum = args.momentum
    activation = args.activation

    # Define the optimizer
    optim = SGD(learning_rate=learning_rate,
                momentum=momentum, nesterov=False)
    if args.optimizer == "adam":
        optim = Adam(learning_rate=learning_rate)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))

    # Add bidirectional LSTM layer
    if args.bidirectional and not args.extra_layer:
        model.add(Bidirectional(LSTM(embedding_dim)))

    # Adds the bidirectional LSTM layers and the possibility to add extra layers, 1 or 2.
    elif args.bidirectional and args.extra_layer:
        model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True)))

        # Adds dropout layer for first layer if asked
        if args.dropout:
            model.add(Dropout(args.dropout))
        if args.extra_layer > 1:
            for i in range(args.extra_layer - 1):
                model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True)))
                # Adds dropout layer for each layer if asked
                if args.dropout:
                    model.add(Dropout(args.dropout))

        model.add(Bidirectional(LSTM(embedding_dim)))

    # Adds the LSTM layers and the possibility to add extra layers, 1 or 2.
    elif args.extra_layer and not args.bidirectional:
        model.add(LSTM(embedding_dim, return_sequences=True))

        # Adds a dropout layer for the first layer if asked
        if args.dropout:
            model.add(Dropout(args.dropout))

        if args.extra_layer > 1:
            for i in range(args.extra_layer - 1):
                model.add(LSTM(embedding_dim, return_sequences=True))

                # Adds dropout layer for each layer if asked
                if args.dropout:
                    model.add(Dropout(args.dropout))
        model.add(LSTM(embedding_dim))

    # Adds the LSTM layer
    else:
        model.add(LSTM(embedding_dim))

    # Adds the last dropout layer if asked
    if args.dropout and args.extra_layer < 2:
        model.add(Dropout(args.dropout))

    # Ultimately, end with dense layer with the activation function
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation=activation))

    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    """Train the LSTM model."""
    verbose = 1
    batch_size = args.batch_size
    epochs = args.epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It"s also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size,
              validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    log_and_print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    log_and_print("f1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))


def predict_transformers(model, tokens_dev, Y_dev_bin, ident):
    """
    Create prediction for the transformer model.
    """
    # Get predictions
    Y_pred = model.predict(tokens_dev)["logits"]
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_dev_bin, axis=1)
    log_and_print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    log_and_print("f1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))


def compile_transformer(lm, args):
    """
    Compile transformer model.
    """
    learning_rate = args.learning_rate
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)
    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    return model


def main():
    """Main function to train and test neural network given cmd line arguments"""
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    if args.transformer:
        if args.transformer == "electra":
            lm = "google/electra-small-discriminator"
        elif args.transformer == "roberta":
            lm = "FacebookAI/roberta-base"
        elif args.transformer == "deberta":
            lm = "microsoft/deberta-v3-base"
        else:
            lm = "distilbert/distilbert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(lm)
        transformer_tokens_train = tokenizer(X_train, padding=True, max_length=100,
                                             truncation=True, return_tensors="np").data
        transformer_tokens_dev = tokenizer(X_dev, padding=True, max_length=100,
                                           truncation=True, return_tensors="np").data

        model = compile_transformer(lm, args)
        model.fit(transformer_tokens_train, Y_train_bin, verbose=1, epochs=args.epochs,
                  batch_size=args.batch_size, validation_data=(transformer_tokens_dev, Y_dev_bin))
        predict_transformers(model, transformer_tokens_dev, Y_dev_bin, "dev")

        # Do predictions on specified test set
        if args.test_file:
            # Read in test set and vectorize
            X_test, Y_test = read_corpus(args.test_file)
            Y_test_bin = encoder.fit_transform(Y_test)
            transformer_tokens_test = tokenizer(X_test, padding=True, max_length=100,
                                                truncation=True, return_tensors="np").data
            # Finally do the predictions
            predict_transformers(model, transformer_tokens_test, Y_test_bin, "test")


    else:
        # Create model
        model = create_model(Y_train, emb_matrix, args)

        # Transform input to vectorized input
        X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
        X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

        # Train the model
        model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, args)

        # Do predictions on specified test set
        if args.test_file:
            # Read in test set and vectorize
            X_test, Y_test = read_corpus(args.test_file)
            Y_test_bin = encoder.fit_transform(Y_test)
            X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
            # Finally do the predictions
            test_set_predict(model, X_test_vect, Y_test_bin, "test")

    # Compile used args for logging.
    all_args = " \\\n".join([f" --{key}={value}" for key, value in vars(args).items() if value])
    log_and_print(f"Used settings:\n{all_args}", False)
    log_and_print(f"Model construction:\n{model.layers}\n", False)


if __name__ == "__main__":
    main()
