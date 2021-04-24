"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
import pandas as pd


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    labels = np.unique(train_labels)
    d, n = train_data.shape
    num_classes = labels.size

    # PRIOR PROBABILITY
    labels, counts = np.unique(train_labels, return_counts=True)
    prior = np.log(counts / sum(counts)).reshape(num_classes, 1)

    # CONDITIONAL PROBABILITY
    conditional = np.zeros((num_classes, d))
    for i in range(num_classes):
        get_class = train_data[:, train_labels == i]
        # For each class, sum up the occur. of each word
        for j in range(d):
            num_words = np.sum(get_class[j, :])
            conditional[i, j] = np.log((num_words + 1) / (num_words.sum(axis=0).reshape(-1, 1) + 2))

    model = {"prior": prior, "conditional": conditional}
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    post_probability = np.matmul(model["conditional"], data) + model["prior"]
    labels = np.argmax(post_probability, axis=0)
    return labels
