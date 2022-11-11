import os
from typing import Callable, Type, Tuple

import numpy as np
from numpy import ndarray

from src.data.agents import QLearningAgent, AgentInterface
from src.data.environments import DynamicForagingTask, EnvironmentInterface
from src.data.generate_synth_data import run_experiment_batch

from src.features.fit_curves import X_BOUNDS, get_sigmoid_feats


def compute_foraging_efficiency(normalized_block_choices: ndarray) -> ndarray:
    """Compute percentage of trials that are correct."""
    if normalized_block_choices.ndim != 2 or normalized_block_choices.shape[1] != X_BOUNDS[1]:
        raise ValueError(f"Input should be of shape (num_blocks, num_trials), and num_trials should be {X_BOUNDS[1]}")
    E = np.mean(normalized_block_choices, axis=1)
    return E


def generate_synth_features(task: Type[EnvironmentInterface],
                            agent: Type[AgentInterface],
                            pr_rew: float,
                            fit_loss: Callable,
                            num_blocks: int = 10) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    # Generate Q learning agent data
    block_choices, param_labels = run_experiment_batch(task,
                                                       agent,
                                                       num_blocks=num_blocks,
                                                       true_pr_rew=pr_rew)
    sigmoid_params = get_sigmoid_feats(block_choices, fit_loss, plot=False)
    feff = compute_foraging_efficiency(block_choices)

    return block_choices, param_labels, sigmoid_params, feff


def normalize_features(feats: ndarray) -> ndarray:
    """
    Normalize features for clustering/classification.
    :param feats: features to normalize, should be of shape (num_samples, num_features).
    :return: standardized features (subtract mean and divide by standard devation across samples for each feature).
    """
    return (feats - np.expand_dims(np.mean(feats, axis=1), 1)) / np.expand_dims(np.std(feats, axis=1), 1)


def remove_invalid_fits(feats: ndarray) -> ndarray:
    """
    Remove samples from an array of sigmoid features that fall outside of predetermined bounds.
    :param feats: Array of sigmoid features.
    :return: Array of sigmoid features with invalid samples removed.
    """
    # invalid_alpha = feats[:, 1] < 0
    invalid_low_s = feats[:, 2] < 0
    invalid_high_s = feats[:, 2] > 14
    invalid_idxs = np.argwhere(invalid_low_s | invalid_high_s)
    return np.delete(feats, invalid_idxs, axis=0)
