from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy import io

from src.data.environments import DynamicForagingTask


class RealSessionDataset:
    """Simplifies access to fields from .mat file."""
    def __init__(self, filename: str) -> None:
        self.data = io.loadmat(filename, simplify_cells=True)['SessionData']

        # Session information
        self.environment = DynamicForagingTask
        self.subject_name = self.data['TrialSettings'][0]['SubjectName']
        self.animal_weight = self.data['AnimalWeight']
        self.ml_water_received = self.data['mLWaterReceived']
        self.num_trials = self.data['nTrials']

        # Basic information about individual trial conditions and outcomes
        self.actions = self.data['ResponseSide']
        self.correct_side = self.data['CorrectSide']
        self.rewarded = self.data['Rewarded']
        self.blocks = self.data['Block']

        # More in-depth information
        self.events = [trial['Events'] for idx, trial in enumerate(self.data['RawEvents']['Trial'])]
        self.states = [trial['States'] for idx, trial in enumerate(self.data['RawEvents']['Trial'])]


def generate_real_block_params(real_blocks: List[int], real_correct_side: List[int]) -> List[Tuple]:
    """
    Generate list of block parameters from real data.
    :param real_blocks: list of ints, where the index is the trial number and entry is the block number, e.g. [1, 1, 1,
    2, 2, 2, 2, 3, 3] would represent 3 trials in block 1, 4 in block 2, and 2 in block 3.
    :param real_correct_side: list of ints, where the index is the trial number and entry is the correct side in the
    block (note this is not necessarily the rewarded side in nondeterministic environments).
    :return: list of block parameters, of format (side, reward_probability, num_trials).
    """
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


def convert_real_actions(real_actions: ndarray) -> ndarray:
    """
    Convert action data from real experiment to correct format (turn nan nonchoices into incorrect choices).
    :param real_actions: List of choices made in experiment, in original format where 1: left choice, 2: right choice,
    nan: no choice.
    :return: List of converted actions, where 0: left choice or no choice
    """
    return np.array([int(act - 1) for act in list(real_actions)])
