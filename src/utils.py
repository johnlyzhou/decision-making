from typing import List, Tuple

import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf, DictConfig


def build_config(env: str,
                 agent: str,
                 num_blocks: int,
                 trial_range: Tuple[int, int],
                 true_p_rew: float,
                 lr: float = None,
                 eps: float = None,
                 pr_switch: float = None,
                 pr_rew: float = None,
                 trans_probs: ndarray = None,
                 save_path: str = None) -> DictConfig:
    """Builds a config file for running an experiment, with an option to save."""
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
    """Ensure that transition matrix is correctly formatted."""
    if len(transition_matrix.shape) != 2:
        raise ValueError("Transition matrix should be 2D!")
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix should be symmetrical!")
    for row in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row]) != 1:
            raise ValueError("Rows of transition probabilities should sum to 1!")


def normalize_choice_block_side(choice_block: List[int],
                                reward_block: List[int] = None,
                                side: str = None,
                                mode: str = 'side',
                                wrong_val: int = 0) -> List[int]:
    """Normalize choice block info to assign the correct choice 1 matching side and the alternative wrong_val. As of
    now, this also labels NaN (no choice) trials as incorrect pending future plans."""
    if mode == 'side':
        if wrong_val == 0:
            if side == "LEFT":
                return [int(choice == 0) for choice in choice_block]
            else:
                return [int(choice == 1) for choice in choice_block]
        else:
            if side == "LEFT":
                return [1 if choice == 0 else wrong_val for choice in choice_block]
            else:
                return [1 if choice == 1 else wrong_val for choice in choice_block]
    elif mode == 'reward':
        return [int(choice_block[i] == reward_block[i]) for i in range(len(choice_block))]
    else:
        raise NotImplementedError


def pad_ragged_blocks(normalized_blocks: List[List[int]], max_len: int = 45) -> ndarray:
    """Take the average choice across blocks of trials with varying lengths."""
    if not max_len:
        max_len = max([len(block) for block in normalized_blocks])
    padded_blocks = np.zeros((max_len, len(normalized_blocks)))
    for idx, block in enumerate(normalized_blocks):
        block = np.array(block)
        lengthened_block = np.pad(block, pad_width=(0, max_len - block.size), mode='constant', constant_values=0)
        padded_blocks[:, idx] = lengthened_block
    return padded_blocks


def truncate_blocks(blocks: List[List[int]], truncate_length: int = 15) -> List[List[int]]:
    """Truncate each block in a list of blocks to a specified length."""
    truncated_blocks = []
    for block in blocks:
        truncated_blocks.append(block[:truncate_length])
    return truncated_blocks


def average_choice_blocks(normalized_choice_blocks: List[List[int]],
                          max_len: int = None,
                          mode: str = 'truncate') -> List[float]:
    """
    Average choices across blocks into a probability of a particular choice for each trial number in the block. Note:
    blocks should already be normalized by correct side, s.t. the correct choice is 1 and incorrect is 0.
    :param normalized_choice_blocks: Normalized choice data, list of blocks of choices.
    :param max_len: Maximum possible length of trials, only need to set if using 'ragged' mode.
    :param mode: Whether to truncate trials to the same length before averaging or do a ragged average.
    :return: Single list of block choices, corresponds to probability of correct choice at a particular trial after
    switch.
    """
    if not max_len:
        max_len = max([len(block) for block in normalized_choice_blocks])

    if mode == 'truncate':
        min_len = min([len(block) for block in normalized_choice_blocks])
        uniform_sum = np.zeros(min_len)
        for block in normalized_choice_blocks:
            block = np.array(block)
            truncated_block = block[:min_len]
            uniform_sum += truncated_block
        return uniform_sum / len(normalized_choice_blocks)
    elif mode == 'ragged':
        ragged_sum = np.zeros(max_len)
        idx_count = np.zeros(max_len)
        for block in normalized_choice_blocks:
            block = np.array(block)
            block_idxs = np.ones(block.size)
            block_idxs = np.pad(block_idxs, pad_width=(0, max_len - block.size), mode='constant', constant_values=0)
            idx_count += block_idxs
            lengthened_block = np.pad(block, pad_width=(0, max_len - block.size), mode='constant', constant_values=0)
            ragged_sum += lengthened_block
        return ragged_sum / idx_count
    else:
        raise NotImplementedError
