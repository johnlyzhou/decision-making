import random

import numpy as np

"""Frequency boundaries separate stimuli that reward left choices from stimuli that reward right choices."""
BOUNDARY_IDX = {'LOW': 0, 'HIGH': 1}
"""Indices in BOUNDARY_IDX map to actual frequency values of the boundaries in BOUNDARY_FREQS"""
BOUNDARY_FREQS = [8, 14]

"""Dictionary keys represent the choice rewarded from the presentation of that stimulus (index)."""
STIMULI_IDXS = {'LEFT': [0], 'SWITCH': [1, 2], 'RIGHT': [3]}
"""The indices in STIMULI_IDXS map to actual frequency values of the stimuli in STIMULI_FREQS."""
STIMULI_FREQS = [6, 10, 12, 16]

"""Actions are choices the mouse can take. In a 2AFC task, it only has 2 choices: left or right)."""
ACTIONS = {'LEFT': 0, 'RIGHT': 1}


def validate_blocks(blocks):
    """
    Ensure block parameters are in the correct format.
    :param blocks: list of tuples, where each tuple represents a block with format (boundary, reward_probability,
    num_trials).
    :return: True if block parameters are in the correct format.
    """
    if not all([len(block) == 3 for block in blocks]):
        raise ValueError("Block parameters should be formatted in a list of tuples (boundary, reward_probability, "
                         "num_trials).")
    for block in blocks:
        boundary, reward_prob, num_trials = block
        if boundary not in BOUNDARY_IDX:
            raise ValueError(f"Boundaries should match one of {BOUNDARY_IDX.keys()}!")
        if reward_prob < 0 or reward_prob > 1.0:
            raise ValueError("Reward probability should be between 0 and 1!")
        if type(num_trials) != int or num_trials < 0:
            raise ValueError("Reward probability should be a positive integer!")
    return True


class Block2AFCTask:
    """2AFC perceptual decision-making task with switching category boundaries from Liu, Xin, and Xu 2021."""
    def __init__(self, blocks, balance_mode="reward"):
        """
        :param blocks: list of tuples, where each tuple represents a block with format (boundary, reward_probability,
        num_trials).
        :param balance_mode: string indicating how probabilities of stimulus appearances should be balanced.
        """
        validate_blocks(blocks)
        self.blocks = blocks
        self.total_trials = sum(block[2] for block in self.blocks)
        self.balance_mode = balance_mode

        self.current_block_idx = 0
        self.current_trial_idx = -1
        self.block_stimulus_schedule = None
        self.block_reward_schedule = None

        self.stimulus_idx_history = []
        self.stimulus_history = []
        self.reward_history = []
        self.boundary_history = []

        for block in blocks:
            self.boundary_history += [block[0] for _ in range(block[2])]

        self.done = False

    def get_current_stimulus(self):
        """Return stimulus presented during current trial."""
        return self.block_stimulus_schedule[self.current_trial_idx]

    def get_current_rewarded_action(self):
        """Return correct choice to receive a reward (not necessarily the one received by the agent)."""
        return self.block_reward_schedule[self.current_trial_idx]

    def step(self):
        """Step through one trial of the experiment."""
        num_trials_in_block = self.blocks[self.current_block_idx][2]

        # Check if environment is finished.
        if self.done is True:
            print("Environment has finished running! Check value of ProposalTaskObject.done to exit any loops.")
            return

        # Check if we've finished the current block.
        if self.current_trial_idx >= num_trials_in_block - 1:
            self.current_trial_idx = -1
            self.current_block_idx += 1

        # Check if we've finished the last block.
        if self.current_block_idx >= len(self.blocks):
            self.end()
            return

        # Check if we're starting a new block: if so, generate new trial schedule.
        if self.current_trial_idx == -1:
            self.block_stimulus_schedule, self.block_reward_schedule = self.sample_schedule()
            self.stimulus_idx_history += list(self.block_stimulus_schedule)
            self.reward_history += list(self.block_reward_schedule)
            self.stimulus_history += [STIMULI_FREQS[stim_idx] for stim_idx in list(self.block_stimulus_schedule)]

        self.current_trial_idx += 1

    def end(self):
        """Set ending flag for environment after all trials in all blocks have finished."""
        self.done = True

    def reset(self):
        """Reset environment to initial state."""
        self.current_block_idx = 0
        self.block_stimulus_schedule = None
        self.block_reward_schedule = None
        self.current_trial_idx = -1
        self.stimulus_idx_history = []
        self.reward_history = []
        self.done = False

    def sample_schedule(self):
        """
        Generate a schedule of trials for the current block, including the stimuli to be presented and the
        corresponding correct choices required to receive a reward.
        :return: two lists of ints of length equal to the number of trials in the block, the first containing indices of
        stimuli presented, the second containing which action will be rewarded, in {0, 1} with 0: left and 1: right.
        """
        boundary, pr_reward, num_trials = self.blocks[self.current_block_idx]

        left_stimuli_idxs = STIMULI_IDXS['LEFT'].copy()
        right_stimuli_idxs = STIMULI_IDXS['RIGHT'].copy()

        if boundary == 'HIGH':
            left_stimuli_idxs += STIMULI_IDXS['SWITCH']
        elif boundary == 'LOW':
            right_stimuli_idxs = STIMULI_IDXS['SWITCH'] + right_stimuli_idxs

        num_left_stimuli = len(left_stimuli_idxs)
        num_right_stimuli = len(right_stimuli_idxs)

        if self.balance_mode == "reward":
            # Balance trials are based on rewarded choice.
            weights = np.ones_like(STIMULI_FREQS, dtype='float')
            weights[left_stimuli_idxs] = weights[left_stimuli_idxs] * num_right_stimuli / num_left_stimuli
            normalized_weights = weights / np.sum(weights)
            stimuli_schedule = random.choices(range(len(STIMULI_FREQS)), weights=normalized_weights, k=num_trials)

        elif self.balance_mode == "stimulus":
            # Balance trials are based on frequency of stimulus appearance.
            weights = np.ones_like(STIMULI_FREQS, dtype='float')
            stimuli_schedule = random.choices(range(len(STIMULI_FREQS)), weights=weights, k=num_trials)

        else:
            raise NotImplemented("Only 'reward' or 'stimulus' balance modes accepted.")

        label_schedule = [stimuli in right_stimuli_idxs for stimuli in stimuli_schedule]
        reward_schedule = np.zeros_like(stimuli_schedule)

        for trial_idx in range(num_trials):
            # We reward the correct choice with pr_reward, and randomly choose one to reward with 1 - pr_reward.
            if np.random.random() < pr_reward:
                reward_schedule[trial_idx] = label_schedule[trial_idx]
            else:
                # Limited to 2AFC tasks.
                if np.random.random() >= 0.5:
                    reward_schedule[trial_idx] = label_schedule[trial_idx]
                else:
                    reward_schedule[trial_idx] = 1 - label_schedule[trial_idx]

        return stimuli_schedule, reward_schedule
