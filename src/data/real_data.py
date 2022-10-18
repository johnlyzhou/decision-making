from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy import io

from src.data.environments import DynamicForagingTask, SwitchingStimulusTask


class DynamicForagingData:
    """Simplifies access to fields from .mat file."""
    def __init__(self, filename: str) -> None:
        self.environment = DynamicForagingTask
        data = io.loadmat(filename)['SessionData'][0][0]
        self.trial_settings = data['TrialSettings'][0]
        self.stages = [setting[8][0] for setting in self.trial_settings]
        self.num_trials = data['nTrials'].flatten()[0]
        self.rewarded = data['Rewarded'].flatten()
        self.animal_weight = data['AnimalWeight'].flatten()[0]
        self.trial_start_time = data['TrialStartTimestamp'][0]
        self.trial_end_time = data['TrialEndTimestamp'][0]
        self.punished = data['Punished'].flatten()
        self.did_not_choose = data['DidNotChoose'].flatten()
        self.iti_jitter = data['ITIjitter'].flatten()
        self.correct_side = data['CorrectSide'].flatten()
        self.decision_gap = data['decisionGap'].flatten()
        self.block = data['Block'].flatten()
        self.assisted = data['Assisted'].flatten()
        self.single_spout = data['SingleSpout'].flatten()
        self.auto_reward = data['AutoReward'].flatten()
        self.response_side = data['ResponseSide'].flatten()
        self.ml_water_received = data['mLWaterReceived'].flatten()[0]


class SwitchingStimulusData:
    """Simplifies access to fields from .mat file."""
    def __init__(self, filename):
        self.environment = SwitchingStimulusTask
        data = io.loadmat(filename)['SessionData'][0][0]
        self.trial_settings = data['TrialSettings'][0]
        self.stages = [setting[8][0] for setting in self.trial_settings]
        self.left_stims = [setting[9][0] for setting in self.trial_settings]
        self.switching_stims = [setting[10][0] for setting in self.trial_settings]
        self.right_stims = [setting[11][0] for setting in self.trial_settings]
        self.num_trials = data['nTrials'].flatten()[0]
        self.rewarded = data['Rewarded'].flatten()
        self.animal_weight = data['AnimalWeight'].flatten()[0]
        self.trial_start_time = data['TrialStartTimestamp'][0]
        self.trial_end_time = data['TrialEndTimestamp'][0]
        self.punished = data['Punished'].flatten()
        self.did_not_choose = data['DidNotChoose'].flatten()
        self.trial_stimulus = data['TrialStimulus'].flatten()
        self.iti_jitter = data['ITIjitter'].flatten()
        self.correct_side = data['CorrectSide'].flatten()
        self.stimulus_duration = data['stimDur'].flatten()
        self.decision_gap = data['decisionGap'].flatten()
        self.stimulus_on = data['stimOn'].flatten()
        self.block = data['Block'].flatten()
        self.decision_threshold = [d[0] for d in data['DecisionThreshold'].flatten()]
        self.assisted = data['Assisted'].flatten()
        self.single_spout = data['SingleSpout'].flatten()
        self.auto_reward = data['AutoReward'].flatten()
        self.response_side = data['ResponseSide'].flatten()
        self.ml_water_received = data['mLWaterReceived'].flatten()[0]
        self.performance = data['Performance'].flatten()
        self.left_performance = data['lPerformance'].flatten()
        self.right_performance = data['rPerformance'].flatten()


def normalize_block_side(action_block: List[int], side: str) -> List[int]:
    """Switch block to make choice matching the hidden state of the trial 1 and the alternative -1."""
    if side == "LEFT":
        return [int(action == 0) for action in action_block]
    else:
        return [int(action == 1) for action in action_block]


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


def convert_real_blocks(real_blocks: List[int], real_correct_side: List[int]) -> List[Tuple]:
    real_blocks = np.array(real_blocks)
    unique, counts = np.unique(real_blocks, return_counts=True)
    blocks = []
    if real_correct_side[0] == 1:
        side = "LEFT"
    else:
        side = "RIGHT"
    for num_trials in counts:
        blocks.append((side, 1.0, num_trials))
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
