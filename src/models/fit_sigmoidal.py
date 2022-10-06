import math

from scipy.optimize import minimize


def epsilon_logistic(x, eps, k, x_0):
    """
    Logistic curve with additional epsilon "lapse" parameter.
    :param eps: float in range [0, 1], epsilon exploration parameter, influences maximum value.
    :param k: logistic growth rate indicating steepness of the curve.
    :param x: int indicating trial index, where the first trial after the block switch has index 0.
    :param x_0: trial index at curve's midpoint.
    :return: float in range [0, 1] indicating some percentage.
    """
    return eps + (1 - 2 * eps) / (1 + math.exp(-k(x - x_0)))
