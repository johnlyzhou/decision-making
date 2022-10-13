from typing import Union, List

import numpy as np
from numpy import ndarray


def build_observations(*args: list) -> ndarray:
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
