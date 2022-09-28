"""This module checks basic functionality of the environments and agents, not designed to be rigorous or independent."""

import unittest

from src.data.agents import QLearningAgent, BeliefStateAgent
from src.data.environments import Block2AFCTask, run_experiment

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
        blocks = [('LOW', 1.0, 100), ('HIGH', 1.0, 100), ('LOW', 1.0, 100), ('HIGH', 1.0, 100)]
        env = Block2AFCTask(blocks)
        for ep in range(sum(block[2] for block in blocks) + 1):
            env.step()
            if env.done:
                break
        for block_idx in range(len(blocks)):
            self.assertEqual(len(env.reward_history[block_idx]),
                             len(env.stimulus_history[block_idx]),
                             blocks[block_idx][2]
                             )


class TestAgents(unittest.TestCase):
    def test_q_learning_agent(self):
        test_task = Block2AFCTask(TEST_BLOCKS)
        learning_rate = 1.0
        epsilon = 0.05
        mf_agent = QLearningAgent(learning_rate, epsilon)
        run_experiment(test_task, mf_agent)
        self.assertEqual(len(mf_agent.action_history),
                         len(mf_agent.stimulus_action_value_history),
                         test_task.total_trials)

    def test_belief_state_agent(self):
        test_task = Block2AFCTask(TEST_BLOCKS)
        p_rew = 1.0
        p_switch = 0.1
        mb_agent = BeliefStateAgent(p_rew, p_switch)
        run_experiment(test_task, mb_agent)
        self.assertEqual(len(mb_agent.action_history),
                         len(mb_agent.boundary_belief_history),
                         test_task.total_trials)


if __name__ == "__main__":
    unittest.main()
