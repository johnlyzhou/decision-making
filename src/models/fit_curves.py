from typing import List

import numpy as np
from numpy import ndarray


def driver_func(params, X, y) -> ndarray:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    """
    yhat = sigmoid(X, *params)
    loss = np.sum((yhat - y) ** 2)
    return loss


def sigmoid(x, eps: float, alpha: float, s: int) -> float:
    """
    Sigmoid curve with additional epsilon "lapse" parameter.
    :param eps: float in range [0, 1], epsilon exploration parameter, influences maximum value.
    :param alpha: logistic growth rate indicating steepness of the curve.
    :param x: int indicating trial index, where the first trial after the block switch has index 0.
    :param s: trial index at curve's midpoint.
    :return: float in range [0, 1] indicating some percentage.
    """
    return 2 * (eps + (1 - 2 * eps) / (1 + np.exp(-alpha * (x - s)))) - 1


def normalize_block_ala_le2022(action_block: List[int], side: str) -> List[int]:
    """Switch block to make choice matching the hidden state of the trial 1 and the alternative -1."""
    if side == "LEFT":
        return [1 if action == 0 else -1 for action in action_block]
    else:
        return [1 if action == 1 else -1 for action in action_block]


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
