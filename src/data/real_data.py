from typing import List, Tuple

import numpy as np
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


def convert_real_actions(real_actions: List[int]) -> List[int]:
    """
    Convert action data from real experiment to correct format (turn nan nonchoices into incorrect choices).
    :param real_actions: List of choices made in experiment, in original format where 1: left choice, 2: right choice,
    nan: no choice.
    :return: List of converted actions, where 0: left choice or no choice
    """
    for i, act in enumerate(real_actions):
        if np.isnan(act):
            real_actions[i] = 0
    return [int(action - 1) for action in real_actions]
