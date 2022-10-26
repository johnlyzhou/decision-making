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


def normalize_block_side(action_block: List[int], side: str, wrong_val: int = 0) -> List[int]:
    """Switch block to make choice matching the hidden state of the trial 1 and the alternative -1."""
    if wrong_val == 0:
        if side == "LEFT":
            return [int(action == 0) for action in action_block]
        else:
            return [int(action == 1) for action in action_block]
    else:
        if side == "LEFT":
            return [1 if action == 0 else wrong_val for action in action_block]
        else:
            return [1 if action == 1 else wrong_val for action in action_block]


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
    truncated_blocks = []
    for block in blocks:
        truncated_blocks.append(block[:truncate_length])
    return truncated_blocks


def average_blocks(normalized_blocks: List[List[int]], max_len: int = 45, mode: str = 'ragged') -> List[float]:
    """Take the average choice across blocks of trials with varying lengths."""
    if not max_len:
        max_len = max([len(block) for block in normalized_blocks])
    if mode == 'ragged':
        ragged_sum = np.zeros(max_len)
        idx_count = np.zeros(max_len)
        for block in normalized_blocks:
            block = np.array(block)
            block_idxs = np.ones(block.size)
            block_idxs = np.pad(block_idxs, pad_width=(0, max_len - block.size), mode='constant', constant_values=0)
            idx_count += block_idxs
            lengthened_block = np.pad(block, pad_width=(0, max_len - block.size), mode='constant', constant_values=0)
            ragged_sum += lengthened_block
        return ragged_sum / idx_count
    else:
        min_len = min([len(block) for block in normalized_blocks])
        uniform_sum = np.zeros(min_len)
        for block in normalized_blocks:
            block = np.array(block)
            truncated_block = block[:min_len]
            uniform_sum += truncated_block
        return uniform_sum / len(normalized_blocks)


def convert_real_blocks(real_blocks: List[int], real_correct_side: List[int]) -> List[Tuple]:
    real_blocks = np.array(real_blocks)
    unique, counts = np.unique(real_blocks, return_counts=True)
    blocks = []
    if real_correct_side[0] == 1:
        side = "LEFT"
    else:
        side = "RIGHT"
    for num_trials in counts:
        blocks.append((side, 1.0, int(num_trials)))
        if side == "LEFT":
            side = "RIGHT"
        else:
            side = "LEFT"
    return blocks


def convert_real_actions(real_actions: List[int]) -> List[int]:
    for i, act in enumerate(real_actions):
        if np.isnan(act):
            real_actions[i] = 1
    return [action - 1 for action in real_actions]


def normalize(feats):
    """Normalize features for clustering/classification"""
    return (feats - np.expand_dims(np.mean(feats, axis=1), 1)) / np.expand_dims(np.std(feats, axis=1), 1)
