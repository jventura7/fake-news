"""
Functions for training and predicting with linear classifiers
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize, check_grad


def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of features.

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size
                    (d, num_classes) ndarray
    :type model: dict
    :return: length n vector of class predictions
    :rtype: array
    """
    y_pred = np.dot(np.transpose(model["weights"]), data)
    label = np.argmax(y_pred, axis=0)
    return label


def log_reg_train(data, labels, model, check_gradient=False):
    """
    Train a linear classifier by maximizing the logistic likelihood (minimizing the negative log logistic likelihood)

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param labels: length n array of the integer class labels
    :type labels: array
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size
                    (d, num_classes) ndarray
    :type model: dict
    :param check_gradient: Boolean value indicating whether to run the numerical gradient check, which will skip
                            learning after checking the gradient on the initial model weights.
    :type check_gradient: Boolean
    :return: the learned model
    :rtype: dict
    """
    n, d = data.shape
    weights = model['weights'].ravel()

    def log_reg_nll(new_weights):
        """
        This internal function returns the negative log-likelihood (nll) as well as the gradient of the nll

        :param new_weights: weights to use for computing logistic regression likelihood
        :type new_weights: ndarray
        :return: tuple containing (<negative log likelihood of data>, gradient)
        :rtype: float
        """
        # reshape the weights, which the optimizer prefers to be a vector, to the more convenient matrix form
        new_weights = new_weights.reshape((n, -1))
        num_classes = np.shape(new_weights)[1]
        # TODO fill in your code here to compute the objective value (nll)
        sum_log = 0
        for j in range(n):
            log_sum_exp = logsumexp(np.dot(np.transpose(new_weights), data[:, j]), 0)
            sum_log = sum_log + log_sum_exp

        sum_second = 0
        for j in range(n):
            # print(j, labels[j], data[:, j])
            dot = np.dot(np.transpose(new_weights[:, labels[j]]), data[:, j])
            sum_second = sum_second + dot

        nll = sum_log - sum_second

        # TODO fill in your code here to compute the gradient fake news how to turn list of strings to
        # compute the gradient
        gradient = np.zeros((n, num_classes))
        mult = np.dot(np.transpose(new_weights), data)
        p_ij = np.transpose(np.exp(mult - logsumexp(mult, 0)))
        for i in range(num_classes):
            calc_gradient = 0
            for j in range(n):
                if labels[j] == i:
                    identifier = 1
                else:
                    identifier = 0
                p = p_ij[j, i] - identifier
                calc_gradient = calc_gradient + (data[:, j] * p)
            gradient[:, i] = calc_gradient

        return nll, gradient

    if check_gradient:
        grad_error = check_grad(lambda w: log_reg_nll(w)[0], lambda w: log_reg_nll(w)[1].ravel(), weights)
        print(
            "Provided gradient differed from numerical approximation by %e (should be around 1e-3 or less)" % grad_error)
        return model

    # pass the internal objective function into the optimizer
    res = minimize(lambda w: log_reg_nll(w)[0], jac=lambda w: log_reg_nll(w)[1].ravel(), x0=weights, method='BFGS')
    weights = res.x

    model = {'weights': weights.reshape((d, -1))}

    return model


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.

    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val

