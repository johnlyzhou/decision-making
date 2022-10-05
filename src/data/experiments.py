from omegaconf import OmegaConf
import numpy as np

from src.data.agents import QLearningAgent, BeliefStateAgent, SwitchingAgent, BlockSwitchingAgent
from src.data.environments import Block2AFCTask
from src.visualization.visualize_expt import plot_psychometric_curve


class Experiment:
    def __init__(self, config):
        """
        :param config: string containing path to config file.
        """
        self.config = OmegaConf.load(config)

        if self.config["environment"].lower() == "block2afctask":
            self.blocks = self.config["blocks"]
            self.environment = Block2AFCTask(self.blocks)
        else:
            raise NotImplementedError("Block2AFCTask is the only implemented environment!")

        if self.config["agent"].lower() == "qlearningagent":
            self.learning_rate = self.config["learning_rate"]
            self.epsilon = self.config["epsilon"]
            if not (0 <= self.epsilon <= 1):
                raise ValueError("Epsilon should be between 0 and 1!")

            self.agent = QLearningAgent(self.learning_rate, self.epsilon)

        elif self.config["agent"].lower() == "beliefstateagent":
            self.p_reward = self.config["p_reward"]
            self.p_switch = self.config["p_switch"]
            if not (0 <= self.p_reward <= 1 and 0 <= self.p_switch <= 1):
                raise ValueError("p_reward and p_switch should be between 0 and 1!")

            self.agent = BeliefStateAgent(self.p_reward, self.p_switch)

        elif self.config["agent"].lower() == "switchingagent" or self.config["agent"].lower() == "blockswitchingagent":
            self.transition_matrix = np.array(self.config["transition_probs"])
            self.learning_rate = self.config["learning_rate"]
            self.epsilon = self.config["epsilon"]
            if not (0 <= self.epsilon <= 1):
                raise ValueError("Epsilon should be between 0 and 1!")
            self.p_reward = self.config["p_reward"]
            self.p_switch = self.config["p_switch"]
            if not (0 <= self.p_reward <= 1 and 0 <= self.p_switch <= 1):
                raise ValueError("p_reward and p_switch should be between 0 and 1!")

            agent0 = QLearningAgent(self.learning_rate, self.epsilon)
            agent1 = self.agent = BeliefStateAgent(self.p_reward, self.p_switch)
            agents = [agent0, agent1]
            if self.config["agent"].lower() == "switchingagent":
                self.agent = SwitchingAgent(self.transition_matrix, agents)
            elif self.config["agent"].lower() == "blockswitchingagent":
                self.agent = BlockSwitchingAgent(self.transition_matrix, agents)
        else:
            raise NotImplementedError("QLearningAgent and BeliefStateAgent are the only implemented agents!")

    def run(self):
        """Run an agent through an experiment with parameters set by block_params."""
        for ep in range(self.environment.total_trials + 1):
            self.environment.step()
            if self.environment.done:
                break
            stimuli = self.environment.get_current_stimuli()
            agent_action = self.agent.sample_action(stimuli)
            correct_action = self.environment.get_current_reward()
            reward = (agent_action == correct_action)
            if self.environment.current_trial == 0 and type(self.agent) == BlockSwitchingAgent:
                self.agent.update(stimuli, agent_action, reward, block_switch=True)
            else:
                self.agent.update(stimuli, agent_action, reward)

    def plot_psychometric_scatter(self, save=False, path=None):
        """
        :param save: boolean indicating whether to save psychometric scatter plot.
        :param path: string indicating file path in which figure will be saved.
        """
        if not self.environment.done:
            raise Exception("Run experiment before plotting results!")

        blocks = self.blocks
        actions = self.agent.action_history
        stimuli = self.environment.stimulus_idx_history
        rewards = self.environment.reward_history

        fig = plot_psychometric_curve(blocks, rewards, stimuli, actions)
        if save and path:
            fig.savefig(path)
        elif save:
            raise ValueError("Save location not specified!")
