from src.agents import QLearningAgent, BeliefStateAgent
from src.block_2afc_task import Block2AFCTask, run_experiment
from src.visualize import plot_psychometric_curve

test_blocks = [('LOW', 1.0, 100), ('HIGH', 1.0, 100)]

test_task = Block2AFCTask(test_blocks)

learning_rate = 1.0
epsilon = 0.05
mf_agent = QLearningAgent(learning_rate, epsilon)

p_rew = 1.0
p_switch = 0.1
mb_agent = BeliefStateAgent(p_rew, p_switch)
print(mb_agent.reward_distribution)
env, agent = run_experiment(test_task, mb_agent, test_blocks)

actions = agent.action_history
stimuli = test_task.stimulus_history
rewards = test_task.reward_history

fig = plot_psychometric_curve(test_blocks, rewards, stimuli, actions)
