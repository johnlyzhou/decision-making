import numpy as np
from numpy import ndarray

from src.features.fit_curves import X_BOUNDS


def compute_foraging_efficiency(normalized_block_choices: ndarray, save_path=None) -> ndarray:
    """Compute percentage of trials that are correct."""
    if normalized_block_choices.ndim != 2 or normalized_block_choices.shape[1] != X_BOUNDS[1]:
        raise ValueError(f"Input should be of shape (num_blocks, num_trials), and num_trials should be {X_BOUNDS[1]}")
    E = np.mean(normalized_block_choices, axis=1)
    if save_path:
        np.save(save_path, E)
    return E
