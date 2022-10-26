from typing import Union, List, Tuple

import numpy as np
from numpy import ndarray


def build_ssm_observations(*args: list) -> ndarray:
    """
    Takes in lists of observations and reshapes them into the correct observation format for the SSM library.
    :param args: lists of observation values of length n (corresponding to trials, blocks, etc.).
    :return: list of tuples with each type of observation of length n.
    """
    if len(set([len(arg) for arg in args])) != 1:
        print([len(arg) for arg in args])
        raise ValueError("All lists of observations should be the same length!")

    obs = list(zip(*args))
    return np.array(obs)


def compute_foraging_efficiency(actions: Union[List[int], ndarray], rewards: Union[List[int], ndarray]) -> float:
    """Compute percentage of trials correct across an entire experiment."""
    if type(actions) is list:
        actions = np.array(actions)
    if type(rewards) is list:
        rewards = np.array(rewards)
    return np.sum((actions == rewards)) / len(actions)


def sigmoid_initial_guess(y: List[float]) -> ndarray:
    """
    Guess initial parameters for sigmoid fitting.
    :param y: trial-averaged decision values for each timestep of the trial
    :return: set of initial parameters
    """
    s = 14
    for i in range(len(y) - 1):
        if y[i] > 0.5:
            s = i
            break
    eps = 0.2
    a = 5
    return np.array([eps, a, s])


def driver_func(params, X, y) -> ndarray:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    """
    yhat = sigmoid(X, *params)
    loss = np.sum((yhat - y) ** 2)
    return loss


def sigmoid(X: ndarray, eps: float, alpha: float, s: int) -> ndarray:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    :param eps: float in range [0, 1], epsilon exploration parameter, influences maximum value.
    :param alpha: logistic growth rate indicating steepness of the curve.
    :param X: int indicating trial index, where the first trial after the block switch has index 0.
    :param s: trial index at curve's midpoint.
    :return: float in range [0, 1] indicating some percentage.
    """
    return np.float64(eps + (1 - 2 * eps) / (1 + np.exp(-alpha * (X - s))))
