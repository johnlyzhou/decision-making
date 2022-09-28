from collections import OrderedDict
import random

import numpy as np

"""Frequency boundaries separate stimuli that reward left choices from stimuli that reward right choices."""
BOUNDARY_IDX = OrderedDict()
BOUNDARY_IDX['LOW'] = 0
BOUNDARY_IDX['HIGH'] = 1
"""Indices in BOUNDARY_IDX map to actual frequency values of the boundaries in BOUNDARY_FREQS"""
BOUNDARY_FREQS = [8, 14]

"""Dictionary keys represent the choice rewarded from the presentation of that stimulus (index)."""
STIMULI_IDXS = OrderedDict()
STIMULI_IDXS['LEFT'] = [0]
STIMULI_IDXS['SWITCH'] = [1, 2]
STIMULI_IDXS['RIGHT'] = [3]
"""The indices in STIMULI_IDXS map to actual frequency values of the stimuli in STIMULI_FREQS."""
STIMULI_FREQS = [6, 10, 12, 16]

"""Actions are choices the mouse can take. In a 2AFC task, it only has 2 choices: left or right)."""
ACTIONS = OrderedDict()
ACTIONS['LEFT'] = 0
ACTIONS['RIGHT'] = 1


class Switch2AFCTask:
    """2AFC perceptual decision-making task from Liu, Xin, and Xu ."""
    def __init__(self, blocks, balance_mode="reward"):
        self.validate_blocks(blocks)

        self.blocks = blocks
        self.balance_mode = balance_mode
        self.current_block = 0
        self.block_stimulus_schedule = None
        self.block_reward_schedule = None
        self.current_trial = -1
        self.current_stimulus = None
        self.current_reward = None

        self.stimulus_history = []
        self.reward_history = []

        self.done = False

    @staticmethod
    def validate_blocks(blocks):
        """Ensure block parameters are in the correct format."""
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

    def get_current_stimuli(self):
        """Return stimulus from current trial."""
        return self.current_stimulus

    def get_current_reward(self):
        """Return ground truth reward choice from current trial (not the one received by the agent)."""
        return self.current_reward

    def step(self):
        """Step through one trial of the experiment."""
        num_trials_in_block = self.blocks[self.current_block][2]

        # Check if environment is finished.
        if self.done is True:
            print("Environment has finished running! Check value of ProposalTaskObject.done to exit any loops.")
            return

        # Check if we've finished the current block.
        if self.current_trial >= num_trials_in_block - 1:
            self.current_trial = -1
            self.current_block += 1

        # Check if we've finished the last block.
        if self.current_block >= len(self.blocks):
            self.end()
            return

        # Check if we're starting a new block - if so, generate new trial schedule.
        if self.current_trial == -1:
            self.block_stimulus_schedule, self.block_reward_schedule = self.sample_schedule()
            self.stimulus_history.append(self.block_stimulus_schedule)
            self.reward_history.append(self.block_reward_schedule)

        self.current_trial += 1
        self.current_stimulus = self.block_stimulus_schedule[self.current_trial]
        self.current_reward = self.block_reward_schedule[self.current_trial]

    def end(self):
        """Set ending flag for environment object."""
        print("Simulation finished!")
        self.done = True

    def sample_schedule(self):
        """Generate a schedule of trials for the current block."""
        boundary, pr_reward, num_trials = self.blocks[self.current_block]

        left_stim_idxs = STIMULI_IDXS['LEFT']
        right_stim_idxs = STIMULI_IDXS['RIGHT']

        if boundary == 'HIGH':
            left_stim_idxs += STIMULI_IDXS['SWITCH']
        elif boundary == 'LOW':
            right_stim_idxs = STIMULI_IDXS['SWITCH'] + right_stim_idxs

        num_left_stimuli = len(left_stim_idxs)
        num_right_stimuli = len(right_stim_idxs)

        if self.balance_mode == "reward":
            # Balance trials are based on rewarded choice.
            weights = np.ones_like(STIMULI_FREQS, dtype='float')
            weights[left_stim_idxs] = weights[left_stim_idxs] * num_right_stimuli / num_left_stimuli
            normalized_weights = weights / np.sum(weights)
            stimuli_schedule = random.choices(range(len(STIMULI_FREQS)), weights=normalized_weights, k=num_trials)

        elif self.balance_mode == "stimulus":
            # Balance trials are based on frequency of stimulus appearance.
            weights = np.ones_like(STIMULI_FREQS, dtype='float')
            stimuli_schedule = random.choices(range(len(STIMULI_FREQS)), weights=weights, k=num_trials)

        else:
            raise NotImplemented("Only 'reward' or 'stimulus' balance modes accepted.")

        label_schedule = [stimuli in right_stim_idxs for stimuli in stimuli_schedule]
        reward_schedule = np.zeros_like(stimuli_schedule)

        for trial_idx in range(num_trials):
            # We reward the correct choice with pr_reward, and randomly choose one to reward with 1-pr_reward.
            if np.random.random() < pr_reward:
                reward_schedule[trial_idx] = label_schedule[trial_idx]
            else:
                # Limited to 2AFC tasks.
                if np.random.random() >= 0.5:
                    reward_schedule[trial_idx] = label_schedule[trial_idx]
                else:
                    reward_schedule[trial_idx] = 1 - label_schedule[trial_idx]

        return stimuli_schedule, reward_schedule


def run_env(env, agent, block_params):
    """Run an agent through and environment with parameters set by block_params."""
    for ep in range(sum(block[2] for block in block_params)):
        env.step()
        if env.done:
            break
        stimuli = env.get_current_stimuli()
        action = agent.sample_action(stimuli)
        rewarded_action = env.get_current_reward()
        reward = (action == rewarded_action)
        agent.update(stimuli, action, reward)

    return env, agent
