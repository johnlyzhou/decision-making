from src.agents import QLearningAgent
from src.environments import (ProposalTask, run_env, LEFT, RIGHT, ALWAYS_LEFT_STIMULI_IDXS, ALWAYS_RIGHT_STIMULI_IDXS,
                              SWITCHING_STIMULI_IDXS, STIMULI)
from src.visualize import plot_psychometric_curve

test_blocks = [(LEFT, 1.0, 100), (RIGHT, 1.0, 100), (LEFT, 1.0, 100), (RIGHT, 1.0, 100)]
num_stimuli = len(ALWAYS_LEFT_STIMULI_IDXS) + len(ALWAYS_RIGHT_STIMULI_IDXS) + len(SWITCHING_STIMULI_IDXS)
assert(num_stimuli == len(STIMULI))

test_task = ProposalTask(test_blocks)
num_states = num_stimuli
num_actions = 2
learning_rate = 0.1
epsilon = 0.1
new_agent = QLearningAgent(num_states, num_actions, learning_rate, epsilon)

env, agent = run_env(test_task, new_agent, test_blocks)

actions = agent.action_history
stimuli = test_task.stimuli_history
rewards = test_task.reward_history

fig = plot_psychometric_curve(test_blocks, rewards, stimuli, actions)
