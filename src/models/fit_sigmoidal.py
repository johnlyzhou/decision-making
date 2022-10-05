import math

from scipy.optimize import minimize


def sigmoidal(t, eps, alpha, s):
    return eps + (1 - 2 * eps) / (1 + math.exp(-alpha(t - s)))
