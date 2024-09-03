#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-ts", "--test_file", default='test.txt', type=str,
                        help="Test file to test the system prediction quality")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    '''TODO: add function description'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_dev, Y_dev = read_corpus(args.dev_file, args.sentiment)
    X_test, Y_test = read_corpus(args.test_file, args.sentiment)

    # Makes a list of unique ordered labels
    unique_labels = []
    for label in Y_test:
        if label not in unique_labels:
            unique_labels.append(label)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    # TODO: comment this
    classifier.fit(X_train, Y_train)

    # TODO: IT USES THE TEST DATA WHILE WE HAVE A DEV SET AS WELL!!!!!
    Y_pred = classifier.predict(X_test)

    # Calculates the accuracy score and the f1 score using sklearn
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="macro")

    # Calculates the confusion matrix using sklearn
    cm = confusion_matrix(Y_test, Y_pred, labels=unique_labels)

    # Calculates the other metrics.
    metrics = precision_recall_fscore_support(Y_test, Y_pred, labels=unique_labels)

    # TODO: Felt like this didn't need to be a function as it is just printing, if you think it needs to be a function we can fix that :)
    for i in range(len(unique_labels)):
        print(f"{unique_labels[i]} precision: {round(metrics[0][i], 3)}")
        print(f"{unique_labels[i]} recall: {round(metrics[1][i], 3)}")
        print(f"{unique_labels[i]} f1-score: {round(metrics[2][i], 3)} \n")

    print(f"The Confusion Matrix: \n {cm} \n")
    print(f"Macro Averaged f1-score: {round(f1, 3)}")
    print(f"Final accuracy: {acc}")
