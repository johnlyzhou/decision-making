"""This module checks basic functionality of the environments and agents, not designed to be rigorous or independent."""

import unittest

from src.data.environments import Block2AFCTask

TEST_BLOCKS = [('LOW', 1.0, 100), ('HIGH', 1.0, 100), ('LOW', 1.0, 100), ('HIGH', 1.0, 100)]


class TestSwitch2AFCTask(unittest.TestCase):
    def test_run(self):
        env = Block2AFCTask(TEST_BLOCKS)
        for ep in range(sum(block[2] for block in TEST_BLOCKS) + 1):
            env.step()
            if env.done:
                break
        self.assertTrue(env.done)

    def test_trial_counts(self):
        env = Block2AFCTask(TEST_BLOCKS)
        for ep in range(sum(block[2] for block in TEST_BLOCKS) + 1):
            env.step()
            if env.done:
                break
        assert(len(env.reward_history) == len(env.stimulus_history) == len(env.boundary_history))


if __name__ == "__main__":
    unittest.main()
