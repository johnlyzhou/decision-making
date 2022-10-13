from typing import List, Tuple

import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf, DictConfig


def build_config(env: str, agent: str, num_blocks: int, trial_range: Tuple[int, int], true_p_rew: float,
                 pr_rew: float = None, pr_switch: float = None, lr: float = None, eps: float = None,
                 trans_probs: ndarray = None, save_path: str = None) -> DictConfig:
    blocks = []
    sides = ["LEFT", "RIGHT"]
    for idx in range(num_blocks):
        num_trials = np.random.randint(trial_range[0], trial_range[1])
        blocks.append((sides[idx % 2], true_p_rew, num_trials))
    base_config = OmegaConf.create({
        "environment": env,
        "agent": agent,
        "learning_rate": lr,
        "epsilon": eps,
        "p_reward": pr_rew,
        "p_switch": pr_switch,
        "transition_probs": trans_probs,
        "blocks": blocks,
        "save_location": save_path,
    })

    if save_path:
        OmegaConf.save(base_config, save_path)

    return base_config


def get_block_indices(blocks: List[Tuple[str, float, int]]) -> List[Tuple[int, int]]:
    """Return list of indices of the first and last trials of each block within all trials of the experiment."""
    indices = []
    trial_idx = 0
    for i in range(len(blocks)):
        start = trial_idx
        end = trial_idx + blocks[i][2]
        indices.append((start, end))
        trial_idx = end
    return indices


def blockify(blocks: List[Tuple[str, float, int]], obs: list) -> List[List[int]]:
    """Partition a list of trial observations into a list of blocks of trial observations."""
    indices = get_block_indices(blocks)
    if sum([block[2] for block in blocks]) != len(obs):
        raise ValueError("Observation length doesn't match block lengths!")
    return [obs[start:end] for start, end in indices]


def validate_transition_matrix(transition_matrix: ndarray) -> None:
    """
    Ensure that transition matrix is correctly formatted.
    :param transition_matrix: symmetrical 2D numpy array containing transition probabilities between strategies/agents.
    """
    if len(transition_matrix.shape) != 2:
        raise ValueError("Transition matrix should be 2D!")
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix should be symmetrical!")
    for row in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row]) != 1:
            raise ValueError("Rows of transition probabilities should sum to 1!")
