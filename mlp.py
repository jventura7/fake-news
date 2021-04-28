"""Functions for training multi-layer perceptrons."""
from __future__ import division
import numpy as np


def logistic(x):
    """
    Sigmoid activation function that returns the activation value and a matrix of the
                derivatives of the elementwise sigmoid operation.

    :param x: ndarray of inputs to be sent to the sigmoid function
    :type x: ndarray
    :return: tuple of (1) activation outputs and (2) the gradients of the
                sigmoid function, both ndarrays of the same shape as x
    :rtype: tuple
    """
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)

    return y, grad


def nll(score, labels):
    """
    Compute the loss function as the negative log-likelihood of the labels by
                                interpreting scores as probabilities.
    :param score: length-n vector of positive-class probabilities
    :type score: array
    :param labels: length-n array of labels in {+1, -1}
    :type labels: array
    :return: tuple containing (1) the scalar negative log-likelihood (NLL) and (2) the length-n gradient of the NLL with respect to the scores.
    :rtype: tuple
    """
    labels01 = np.copy(labels) # create a copy of labels in {0, 1}
    labels01[labels01 < 0] = 0
    
    objective = np.sum(labels01 * np.log(score) - (1 - labels01) * np.log(1 - score))
    
    grad = np.divide((score - labels01), (score * (1 - score)))

    return objective, grad


def mlp_predict(data, model):
    """
    Predict binary-class labels for a batch of test points

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param model: learned model from mlp_train
    :type model: dict
    :return: array of +1 or -1 labels
    :rtype: array
    """
    n = data.shape[1]

    weights = model['weights']
    activation_function = model['activation_function']

    num_layers = len(weights)

    activation_derivatives = []
    # store all activations of neurons. Start with the data augmented with an all-ones feature
    activations = [np.vstack((data, np.ones((1, n))))]

    for layer in range(num_layers):
        new_activations, activation_derivative = activation_function(weights[layer].dot(activations[layer]))
        activations.append(new_activations)
        activation_derivatives.append(activation_derivative)

    scores = activations[-1].ravel()
    labels = 2 * (scores > 0.5) - 1

    return labels, scores, activations, activation_derivatives


def mlp_objective(model, data, labels, loss_function):
    """
    Compute the objective and gradient of multi-layer perceptron

    :param model: dict containing the current model 'weights'
    :type model: dict
    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: length-n array of labels in {+1, -1}
    :type labels: array
    :param loss_function: function that evaluates a vector of estimated
                            positive-class probabilities against a vector of
                            ground-truth labels. Returns a tuple containing the
                            loss and its gradient.
    :type loss_function: function
    :return: tuple containing (1) the scalar loss objective value and (2) the
                                list of gradients for each weight matrix
    :rtype: tuple
    """
    n = labels.size

    # forward propagation
    weights = model['weights']
    num_layers = len(weights)
    _, scores, activations, activation_derivatives = mlp_predict(data, model)

    # back propagation
    layer_delta = [None] * num_layers
    layer_gradients = [None] * num_layers

    # compute objective value and gradient of the loss function
    objective, gradient = loss_function(scores, labels)
    layer_delta[-1] = gradient.reshape((1, -1))

    # back-propagate error to previous layers
    for i in reversed(range(num_layers - 1)):
        layer_delta[i] = weights[i + 1].T.dot((layer_delta[i + 1] * activation_derivatives[i + 1]))

    # use computed layer_delta to compute gradients for each layer
    for i in range(num_layers):
        layer_gradients[i] = (layer_delta[i] * activation_derivatives[i]).dot(activations[i].T)
        
    return objective, layer_gradients


def mlp_train(data, labels, params, model=None):
    """
    Train a multi-layered perceptron (MLP) with gradient descent and
    back-propagation.

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: array of length n whose entries are all +1 or -1
    :type labels: array
    :param params: dictionary containing model parameters, most importantly
                    'num_hidden_units', which is a list
                    containing the number of hidden units in each hidden layer.
    :type params: dict
    :return: learned MLP model containing 'weights' list of each layer's weight
            matrix
    :rtype: dict
    """
    input_dim, num_train = data.shape

    if not model:
        # no initial model was provided. Initialize model based on params
        model = dict()
        num_hidden_units = params['num_hidden_units']
        model['num_hidden_units'] = num_hidden_units
        # store weight matrices for each model layer in a list
        model['weights'] = list()
        # create input layer
        model['weights'].append(0.1 * np.random.randn(num_hidden_units[0], input_dim + 1))
        # create intermediate layers
        for layer in range(1, len(num_hidden_units)):
            model['weights'].append(0.1 * np.random.randn(num_hidden_units[layer], num_hidden_units[layer-1]))
        # create output layer
        model['weights'].append(0.1 * np.random.randn(1, num_hidden_units[-1]))
        model['activation_function'] = params['activation_function']

    loss_function = params['loss_function']
    num_layers = len(model['weights'])
    lam = params['lambda']

    # Begin training

    objective = np.zeros(params['max_iter'])
    for t in range(params['max_iter']):
        # compute the objective and gradient, which is a list of gradients for each layer's weight matrices
        objective[t], grad = mlp_objective(model, data, labels, loss_function)

        rate = 1.0 / np.sqrt(t + 1)

        total_change = 0

        # use the gradient to update each layer's weights
        for j in range(num_layers):
            change = - rate * (grad[j] + lam * model['weights'][j])
            total_change += np.sum(np.abs(change))

            # clip change to [-0.1, 0.1] to avoid numerical errors
            change = np.clip(change, -0.1, 0.1)

            model['weights'][j] += change

        if total_change < 1e-8:
            print("Exiting because total change was %e, a sign that we have reached a local minimum." % total_change)
            # stop if we are no longer changing the weights much
            break

    return model
