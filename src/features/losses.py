from typing import Callable

import numpy as np
from numpy import ndarray

from src.features.fit_curves import epsilon_sigmoid


def mse_loss(params: list, X: ndarray, y: ndarray) -> ndarray:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    """
    yhat = epsilon_sigmoid(X, *params)
    loss = np.sum((yhat - y) ** 2)
    return loss


def binary_nll(params: list, X: ndarray, y: ndarray) -> ndarray:
    """
    Calculate negative log likelihood for binary logistic regression.
    :param params: List of params (b0 and b1) for binary logistic function.
    :param X: Trial indices from block switch.
    :param y: Binary target values in {0, 1}.
    :return: Negative log-likelihood of y given parameters b0 and b1.
    """
    p_hat = epsilon_sigmoid(X, *params)
    return np.sum(np.multiply(-y, np.log(p_hat)) - np.multiply((1 - y), (np.log(1 - p_hat))))
