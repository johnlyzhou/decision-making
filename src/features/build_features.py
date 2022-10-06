import numpy as np


def build_observations(*args):
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
