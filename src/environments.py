import random

import numpy as np

STIMULI = range(4)
ALWAYS_LEFT_STIMULI = [0]
SWITCHING_STIMULI = [1, 2]
ALWAYS_RIGHT_STIMULI = [3]

# Labels
LEFT = 0
RIGHT = 1

# List of tuples (side, Pr(reward), num_trials_in_block)
test_blocks = [(LEFT, 0.9, 50), (RIGHT, 0.9, 50), (LEFT, 0.9, 50), (RIGHT, 0.9, 50)]
num_stimuli = len(ALWAYS_LEFT_STIMULI) + len(ALWAYS_RIGHT_STIMULI) + len(SWITCHING_STIMULI)


def validate_blocks(blocks):
    if not all([True if (1.0 >= block_prob[0] >= 0) else False for block_prob in blocks]):
        raise ValueError("Block probabilities must be between 0 and 1 (inclusive).")


class ProposalTask:
    def __init__(self, blocks):
        validate_blocks(blocks)
        self.blocks = blocks
        self.current_block = 0
        self.block_stimuli_schedule = None
        self.block_reward_schedule = None

        self.current_trial = -1
        self.current_stimuli = None
        self.current_reward = None

        self.done = False

    def step(self):
        num_trials_in_block = self.blocks[self.current_block][2]

        if self.current_trial >= num_trials_in_block - 1:
            self.current_trial = -1
            self.current_block += 1

        if self.current_block >= len(self.blocks):
            self.end()
            return

        if self.current_trial == -1:
            print("Starting block {}".format(self.current_block))
            self.block_stimuli_schedule, self.block_reward_schedule = self.sample_schedule()
            print(list(zip(self.block_stimuli_schedule, self.block_reward_schedule)))

        self.current_trial += 1
        self.current_stimuli = self.block_stimuli_schedule[self.current_trial]
        self.current_reward = self.block_reward_schedule[self.current_trial]

    def end(self):
        # save metadata, schedules
        print("Done")
        self.done = True

    def sample_schedule(self, balance_mode="reward"):
        side, pr_reward, num_trials = self.blocks[self.current_block]

        if side is LEFT:
            left_stimuli = ALWAYS_LEFT_STIMULI + SWITCHING_STIMULI
            right_stimuli = ALWAYS_RIGHT_STIMULI
        else:
            left_stimuli = ALWAYS_LEFT_STIMULI
            right_stimuli = SWITCHING_STIMULI + ALWAYS_RIGHT_STIMULI

        num_left_stimuli = len(left_stimuli)
        num_right_stimuli = len(right_stimuli)

        if balance_mode is "reward":
            weights = np.ones_like(STIMULI, dtype='float')
            weights[left_stimuli] = weights[left_stimuli] * num_right_stimuli / num_left_stimuli
            normalized_weights = weights / np.sum(weights)
            stimuli_schedule = random.choices(STIMULI, weights=normalized_weights, k=num_trials)

        elif balance_mode is "stimulus":
            weights = np.ones_like(STIMULI, dtype='float')
            stimuli_schedule = random.choices(STIMULI, weights=weights, k=num_trials)

        else:
            raise NotImplemented("Only 'reward' or 'stimulus' balance modes accepted.")

        label_schedule = [stimuli in right_stimuli for stimuli in stimuli_schedule]
        reward_schedule = np.zeros_like(stimuli_schedule)
        for trial_idx in range(num_trials):
            if np.random.random() < pr_reward:
                reward_schedule[trial_idx] = label_schedule[trial_idx]
            else:
                if np.random.random() >= 0.5:
                    reward_schedule[trial_idx] = label_schedule[trial_idx]
                else:
                    reward_schedule[trial_idx] = 1 - label_schedule[trial_idx]

        return stimuli_schedule, reward_schedule
