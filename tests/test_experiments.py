"""This module checks basic functionality of the environments and agents, not designed to be rigorous or independent."""

import unittest
from src.data.environments import Block2AFCTask, STIMULI_IDXS, STIMULI_FREQS
from src.data.experiments import Experiment
from src.utils import blockify

cwd = "/Users/johnzhou/research/decision-making"

TEST_BLOCKS = [('LOW', 1.0, 100), ('HIGH', 1.0, 100), ('LOW', 1.0, 100), ('HIGH', 1.0, 100)]


class TestQLearningAgent(unittest.TestCase):
    pass


class TestBlock2AFCTask(unittest.TestCase):
    def test_trial_counts(self):
        env = Block2AFCTask(TEST_BLOCKS, balance_mode="reward")
        for ep in range(sum(block[2] for block in TEST_BLOCKS) + 1):
            env.step()
            if env.done:
                break
        assert (len(env.reward_history) ==
                len(env.stimulus_history) ==
                len(env.boundary_history) ==
                sum(block[2] for block in TEST_BLOCKS))

    def test_sample_schedule(self):
        env = Block2AFCTask(TEST_BLOCKS, balance_mode="reward")
        for ep in range(sum(block[2] for block in TEST_BLOCKS) + 1):
            env.step()
            if env.done:
                break
        env.reset()
        boundary, p_rew, num_trials = TEST_BLOCKS[0]
        stimuli_schedule, reward_schedule = env.sample_schedule()

        left_stimuli_idxs = STIMULI_IDXS['LEFT'].copy()
        right_stimuli_idxs = STIMULI_IDXS['RIGHT'].copy()

        if boundary == 'HIGH':
            left_stimuli_idxs += STIMULI_IDXS['SWITCH']
        elif boundary == 'LOW':
            right_stimuli_idxs = STIMULI_IDXS['SWITCH'] + right_stimuli_idxs

        for trial_idx in range(len(stimuli_schedule)):
            assert (reward_schedule[trial_idx] == (stimuli_schedule[trial_idx] in right_stimuli_idxs))

    def test_run(self):
        env = Block2AFCTask(TEST_BLOCKS, balance_mode="reward")
        for ep in range(sum(block[2] for block in TEST_BLOCKS) + 1):
            env.step()
            if env.done:
                break
        self.assertTrue(env.done)


class TestExperiment(unittest.TestCase):
    def test_rewards(self):
        self.test_config = f"{cwd}/tests/test_config.yaml"
        self.expt = Experiment(self.test_config)
        self.expt.run()
        rewards = blockify(self.expt.blocks, self.expt.environment.reward_history)
        stimuli = blockify(self.expt.blocks, self.expt.environment.stimulus_idx_history)
        for idx, block in enumerate(self.expt.blocks):
            num_trials = block[2]
            left_stimuli_idxs = STIMULI_IDXS['LEFT'].copy()
            right_stimuli_idxs = STIMULI_IDXS['RIGHT'].copy()
            boundary = block[0]
            if boundary == 'HIGH':
                left_stimuli_idxs += STIMULI_IDXS['SWITCH']
                assert (len(left_stimuli_idxs + right_stimuli_idxs) == len(STIMULI_FREQS))
            elif boundary == 'LOW':
                right_stimuli_idxs = STIMULI_IDXS['SWITCH'] + right_stimuli_idxs
                assert (len(left_stimuli_idxs + right_stimuli_idxs) == len(STIMULI_FREQS))
            for trial_idx in range(num_trials):
                assert (rewards[idx][trial_idx] == (stimuli[idx][trial_idx] in right_stimuli_idxs))


if __name__ == "__main__":
    unittest.main()
