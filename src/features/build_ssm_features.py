import numpy as np


def build_outputs(*args):
    """Takes in lists of observations and reshapes them into the correct observation format for the SSM library."""
    if len(set([len(arg) for arg in args])) != 1:
        print([len(arg) for arg in args])
        raise ValueError("All lists of observations should be the same length!")

    obs = list(zip(*args))
    return np.array(obs)
