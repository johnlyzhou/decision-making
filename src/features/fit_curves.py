from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

SIGMOID_PARAM_BOUNDS = ((0, 0.5), (0, 4), (-1, 14))
X_BOUNDS = (0, 15)


def sigmoid_params_initial_guess(y: List[float]) -> ndarray:
    """
    Guess initial parameters for sigmoid fitting.
    :param y: Trial-averaged decision values for each timestep of the trial.
    :return: Set of initial parameters.
    """
    s = 14
    for i in range(len(y) - 1):
        if y[i] > 0.5:
            s = i
            break
    eps = 0.2
    a = 2
    return np.array([eps, a, s])


def epsilon_sigmoid(X: ndarray,
                    eps: float,
                    alpha: float,
                    s: int) -> ndarray:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    :param eps: Float in range [0, 1], epsilon exploration parameter, influences maximum value.
    :param alpha: Logistic growth rate indicating steepness of the curve.
    :param X: Int indicating trial index, where the first trial after the block switch has index 0.
    :param s: Trial index at curve's midpoint.
    :return: Float in range [0, 1] indicating some percentage.
    """
    return eps + (1 - 2 * eps) / (1 + np.exp(-alpha * (X - s)))


def binary_logistic(X: ndarray,
                    b0: float,
                    b1: float) -> float:
    """
    Binary logistic function.
    :param X: Trial indices from block switch.
    :param b0: Bias term.
    :param b1: Weight on X.
    :return: P(y = 1 | X), or probability of a correct choice (after normalization) given number of trial from switch.
    """
    return 1 / (1 + np.exp(-X * b1 - b0))


def get_sigmoid_feats(truncated_actions: Union[ndarray, list],
                      loss: Callable,
                      plot=False) -> ndarray:

    if type(truncated_actions) == ndarray:
        truncated_actions = list(truncated_actions)

    sig_feats = np.zeros((len(truncated_actions), 3))

    for idx, action_block in enumerate(tqdm(truncated_actions)):
        x_obs = np.arange(len(action_block))
        y_obs = action_block

        try:
            params = scipy.optimize.minimize(loss, sigmoid_params_initial_guess(y_obs), args=(x_obs, y_obs),
                                             bounds=SIGMOID_PARAM_BOUNDS).x
            sig_feats[idx, :] = np.array([*params])
        except RuntimeError:
            print("Failed to fit.")
            continue

        if plot:
            plt.plot(np.linspace(*X_BOUNDS, num=1000),
                     epsilon_sigmoid(np.linspace(*X_BOUNDS, num=1000), *params), 'r-',
                     label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(params))
            plt.scatter(range(15), list(action_block))
            plt.xlim(X_BOUNDS)
            plt.ylim([0, 1])
            plt.legend()
            plt.show()

    return sig_feats


def get_logistic_feats(truncated_actions: Union[ndarray, list],
                       plot=False) -> ndarray:

    if type(truncated_actions) == ndarray:
        truncated_actions = list(truncated_actions)
    logistic_feats = np.zeros((len(truncated_actions), 2))

    for idx, action_block in enumerate(tqdm(truncated_actions)):
        X = np.arange(len(action_block)).reshape(-1, 1)
        y = np.array(action_block)

        unique = np.unique(y)

        a = LogisticRegression(random_state=0)

        if unique.size == 1:
            a.coef_ = np.array([[0]])
            a.classes_ = np.array([0, 1])
            if unique[0] == 0:
                a.intercept_ = np.array([-1])
            else:
                a.intercept_ = np.array([1])

        else:
            a.fit(X, y)

        logistic_feats[idx, :] = np.array([a.coef_[0][0], a.intercept_[0]])

        if plot:
            X_plot = np.linspace(0, 15, num=1000)
            plt.scatter(range(15), list(action_block))
            y_plot = a.predict_proba(X_plot.reshape(-1, 1))[:, 1]

            plt.scatter(X_plot, y_plot, s=5)
            plt.xlim([0, 14])
            plt.ylim([0, 1])
            plt.show()

    return logistic_feats
