import random

import numpy as np


LOW_IDX = 0
HIGH_IDX = 1
BOUNDARIES = [8, 14]

ALWAYS_LEFT_STIMULI_IDXS = [0]
SWITCHING_STIMULI_IDXS = [1, 2]
ALWAYS_RIGHT_STIMULI_IDXS = [3]
STIMULI = [6, 10, 12, 16]

LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]


def validate_blocks(blocks):
    if not all([True if (1.0 >= block_prob[0] >= 0) else False for block_prob in blocks]):
        raise ValueError("Block probabilities must be between 0 and 1 (inclusive).")


class ProposalTask:
    def __init__(self, blocks, balance_mode="reward"):
        validate_blocks(blocks)
        self.blocks = blocks
        self.balance_mode = balance_mode
        self.current_block = 0
        self.block_stimuli_schedule = None
        self.block_reward_schedule = None
        self.stimuli_history = []
        self.reward_history = []

        self.current_trial = -1
        self.current_stimuli = None
        self.current_reward = None

        self.done = False

    def get_current_stimuli(self):
        return self.current_stimuli

    def get_current_reward(self):
        return self.current_reward

    def step(self):
        num_trials_in_block = self.blocks[self.current_block][2]

        if self.current_trial >= num_trials_in_block - 1:
            self.current_trial = -1
            self.current_block += 1

        if self.current_block >= len(self.blocks):
            self.end()
            return

        if self.current_trial == -1:
            self.block_stimuli_schedule, self.block_reward_schedule = self.sample_schedule()
            self.stimuli_history.append(self.block_stimuli_schedule)
            self.reward_history.append(self.block_reward_schedule)

        self.current_trial += 1
        self.current_stimuli = self.block_stimuli_schedule[self.current_trial]
        self.current_reward = self.block_reward_schedule[self.current_trial]

    def end(self):
        # save metadata, schedules
        print("Done")
        self.done = True

    def sample_schedule(self):
        side, pr_reward, num_trials = self.blocks[self.current_block]

        if side is LEFT:
            left_stimuli = ALWAYS_LEFT_STIMULI_IDXS + SWITCHING_STIMULI_IDXS
            right_stimuli = ALWAYS_RIGHT_STIMULI_IDXS
        else:
            left_stimuli = ALWAYS_LEFT_STIMULI_IDXS
            right_stimuli = SWITCHING_STIMULI_IDXS + ALWAYS_RIGHT_STIMULI_IDXS

        num_left_stimuli = len(left_stimuli)
        num_right_stimuli = len(right_stimuli)

        if self.balance_mode == "reward":
            weights = np.ones_like(STIMULI, dtype='float')
            weights[left_stimuli] = weights[left_stimuli] * num_right_stimuli / num_left_stimuli
            normalized_weights = weights / np.sum(weights)
            stimuli_schedule = random.choices(range(len(STIMULI)), weights=normalized_weights, k=num_trials)

        elif self.balance_mode == "stimulus":
            weights = np.ones_like(STIMULI, dtype='float')
            stimuli_schedule = random.choices(range(len(STIMULI)), weights=weights, k=num_trials)

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


def run_env(env, agent, block_params):
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
