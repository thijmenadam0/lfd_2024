#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse

from matplotlib import pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def check_valid_gamma(value):
    """
    Check if the given value for gamma is valid.

    Arguments:
        value (str): The given gamma value.

    returns:
        string | float: The given gamma value.
    """
    if value in ['scale', 'auto']:
        return value

    try:
        float_value = float(value)
        if float_value > 0.0:
            return float_value
        else:
            raise argparse.ArgumentTypeError(f"Given gamma: {value} is invalid, should be positive.")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Given gamma: {value} is invalid, should be float or 'scale' or 'auto'.")


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
    parser.add_argument("-vec", "--vectorizer", choices=["bow", "tfidf", "both"],
                        default="bow", help="Select vectorizer: bow (bag of words), tfidf or both")
    parser.add_argument("-cm", "--confusion_matrix", action="store_true",
                        help="Show extended confusion matrix")
    parser.add_argument("-ng", "--ngram_range", nargs=2, type=int, default=(1, 1),
                        help="Set the ngram range, give two integers separated by space")

    parser.add_argument("-a", "--alpha", default=1.0, type=float,
                        help="Set the alpha for the base Naive Bayes classifier")

    # Creating a subparser allows us to set arguments for hyperparameters per algorithm.
    subparser = parser.add_subparsers(dest="algorithm", required=False,
                                      help="Choose the classifying algorithm to use")

    # Subparser for dummy baseline.
    dummy_parser = subparser.add_parser("dummy",
                                        help="Dummy classifier.")
    dummy_parser.add_argument("-ds", "--dummy_strategy", choices=["most_frequent", "prior", "stratified", "uniform"],
                              default="prior", )

    # Subparser for SVM.
    svm_parser = subparser.add_parser("svm",
                                      help="Use Support Vector Machine as classifier")
    svm_parser.add_argument("-c", "--C", default=1.0, type=float,
                            help="Set the regularization parameter C of SVM")
    svm_parser.add_argument("-g", "--gamma", default='scale', type=check_valid_gamma,
                            help="Set gamma value, can be scale, auto or positive float.")
    svm_parser.add_argument("-sh", "--shape", choices=["ovo", "ovr"], default="ovr",
                            help="Set the decision function shape, one-versus-one or one-versus-rest")
    svm_parser.add_argument("-k", "--kernel", choices=["linear", "poly", "rbf", "sigmoid"],
                            default="rbf",
                            help="Set the kernel, linear is already used by Linear SVM")
    svm_parser.add_argument("-dg", "--degree", default=3, type=int,
                            help="Set the degree, only works for poly kernel")

    # Subparser for Linear SVM
    svml_parser = subparser.add_parser("svml",
                                       help="Use Linear kernel Support Vector Machine as classifier")
    svml_parser.add_argument("-c", "--C", default=1.0, type=float,
                             help="Set the regularization parameter C of Linear SVM")
    svml_parser.add_argument("-p", "--penalty", choices=["l1", "l2"], default="l2",
                             help="Set the penalty parameter for Linear SVM")
    svml_parser.add_argument("-l", "--loss", choices=["hinge", "squared_hinge"], default="squared_hinge",
                             help="Set the loss parameter for Linear SVM, using hinge and penalty l1 is not supported "
                                  "by model")

    # Subparser for K-Nearest Neighbors
    knn_parser = subparser.add_parser("knn",
                                      help="Use K-Nearest Neighbours algorithm as classifier")
    knn_parser.add_argument("-n", "--neighbors", default=5, type=int,
                            help="Set the amount of neighbors for the KNN classifier")
    # TODO: Distance seems to be the better weight, but it is not the default.
    knn_parser.add_argument("-w", "--weight", choices=["uniform", "distance"], default="uniform",
                            help="Set the weight function used in the prediction.")
    # 1 chooses the Manhattan distance, 2 chooses the Euclidean distance.
    knn_parser.add_argument("-p", "--distance", choices=[1, 2], default=2, type=int,
                            help="Set the distance metric. 1 is the Manhattan distance, "
                            "2 is the Euclidean distance.")

    # Parent parser containing the overlapping arguments for Decision Tree and Random Forest
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-c", "--criterion", choices=["gini", "entropy", "log_loss"],
                               default="gini", help="Assign the fuction to measure the split quality")
    parent_parser.add_argument("-md", "--max_depth", default=None, type=int,
                               help="Set the maximum depth of the tree")
    parent_parser.add_argument("-mss", "--min_samples_split", default=2, type=int,
                               help="Set the minimum number of samples required to split an internal node")
    parent_parser.add_argument("-msl", "--min_samples_leaf", default=1, type=int,
                               help="Set the minimum number of samples per leaf node")

    # Subparser for Decision Tree Classifier
    tree_parser = subparser.add_parser("dt", parents=[parent_parser],
                                       help="Use Decision Tree algorithm as classifier")
    tree_parser.add_argument("-s", "--splitter", choices=["best", "random"], default="best",
                             help="Set the strategy used to choose the split of each node")

    # Subparser for Random Forest Classifier
    forest_parser = subparser.add_parser("rf", parents=[parent_parser],
                                         help="Use Random Forest algorithm as classifier")
    forest_parser.add_argument("-n", "--n_estimators", default=100, type=int,
                               help="Sets the number of trees in the forest")

    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    """
    Load a given file and create a list of lists of the documents as tokens and a list of labels.
    
    Arguments:
        corpus_file (str): Path to the corpus file to load.
        use_sentiment (bool): Bool whether to use sentiment analysis or not.
    
    Returns:
        list: List of lists of documents as tokens.
        list: List of labels.

    """
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


def show_confusion_matrix(disp):
    """
    Show a detailed confusion matrix.

    Arguments:
        disp (ConfusionMatrixDisplay): The confusion matrix to show.
    """
    disp.plot()
    plt.show()


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)  # use dev set as test set for now
    # X_dev, Y_dev = read_corpus(args.dev_file, args.sentiment)
    # X_test, Y_test = read_corpus(args.test_file, args.sentiment)

    # Makes a list of unique ordered labels
    unique_labels = []
    for label in Y_test:
        if label not in unique_labels:
            unique_labels.append(label)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(args.ngram_range))
    bow = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(args.ngram_range))
    union = FeatureUnion([("count", bow), ("tf", tf_idf)])

    if args.vectorizer == "tfidf":
        # TF-IDF vectorizer
        vec = tf_idf
    elif args.vectorizer == "bow":
        # Bag of Words vectorizer
        vec = bow
    elif args.vectorizer == "both":
        # Use BoW and TF-IDF vectorizers.
        vec = union

    algorithm = MultinomialNB(alpha=args.alpha)

    # Best setup C=3, gamma=scale, decision_shape_function=ovr
    if args.algorithm == "svm":
        algorithm = SVC(C=args.C, gamma=args.gamma, decision_function_shape=args.shape, kernel=args.kernel,
                        degree=args.degree)

    # Best setup C=0.7, penalty=l2, loss=squared_loss
    elif args.algorithm == "svml":
        algorithm = LinearSVC(C=args.C, penalty=args.penalty, loss=args.loss)

    elif args.algorithm == "knn":
        algorithm = KNeighborsClassifier(n_neighbors=args.neighbors, weights=args.weight, p=args.distance)
    elif args.algorithm == "dt":
        algorithm = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth,
                                           min_samples_split=args.min_samples_split,
                                           min_samples_leaf=args.min_samples_leaf)
    elif args.algorithm == "rf":
        algorithm = RandomForestClassifier(n_estimators=args.n_estimators, criterion=args.criterion,
                                           max_depth=args.max_depth, min_samples_split=args.min_samples_split,
                                           min_samples_leaf=args.min_samples_leaf)

    elif args.algorithm == "dummy":
        algorithm = DummyClassifier(strategy=args.dummy_strategy)

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    classifier = Pipeline([('vec', vec), ('cls', algorithm)])

    # TODO: comment this
    classifier.fit(X_train, Y_train)

    # TODO: IT USES THE TEST DATA WHILE WE HAVE A DEV SET AS WELL!!!!!
    Y_pred = classifier.predict(X_test)

    # Calculates the accuracy score and the f1 score using sklearn
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="macro")

    # Calculates the confusion matrix using sklearn
    cm = confusion_matrix(Y_test, Y_pred, labels=unique_labels)

    # Add labels to confusion matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)

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

    if args.confusion_matrix:
        show_confusion_matrix(disp)
