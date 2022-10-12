from typing import Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np

from src.data.agents import QLearningAgent, BeliefStateAgent, SwitchingAgent, BlockSwitchingAgent, InferenceAgent
from src.data.environments import BlockStimulusTask, BlockTask


class Experiment:
    """
    Sets up a task environment and agent(s) and runs an experiment based on the settings in a configuration file.
    """
    def __init__(self, config: Union[str, DictConfig]) -> None:
        """
        :param config: string containing path to config file.
        """
        self.done = False
        if type(config) == str:
            self.config = OmegaConf.load(config)
        else:
            self.config = config
        self.task_type = None

        if self.config["environment"].lower() == "blockstimulustask":
            self.task_type = BlockStimulusTask
            self.blocks = self.config["blocks"]
            self.environment = BlockStimulusTask(self.blocks)
        elif self.config["environment"].lower() == "blocktask":
            self.task_type = BlockTask
            self.blocks = self.config["blocks"]
            self.environment = BlockTask(self.blocks)
        else:
            raise NotImplementedError

        if self.config["agent"].lower() == "qlearningagent":
            self.learning_rate = self.config["learning_rate"]
            self.epsilon = self.config["epsilon"]
            if not (0 <= self.epsilon <= 1):
                raise ValueError("Epsilon should be between 0 and 1!")

            self.agent = QLearningAgent(self.learning_rate, self.epsilon, task=self.task_type)

        elif self.config["agent"].lower() == "inferenceagent":
            self.p_reward = self.config["p_reward"]
            self.p_switch = self.config["p_switch"]
            if not (0 <= self.p_reward <= 1 and 0 <= self.p_switch <= 1):
                raise ValueError("p_reward and p_switch should be between 0 and 1!")

            self.agent = InferenceAgent(self.p_reward, self.p_switch, task=self.task_type)

        elif self.config["agent"].lower() == "beliefstateagent":
            self.p_reward = self.config["p_reward"]
            self.p_switch = self.config["p_switch"]
            if not (0 <= self.p_reward <= 1 and 0 <= self.p_switch <= 1):
                raise ValueError("p_reward and p_switch should be between 0 and 1!")

            self.agent = BeliefStateAgent(self.p_reward, self.p_switch, task=self.task_type)

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

            agent0 = QLearningAgent(self.learning_rate, self.epsilon, task=self.task_type)
            agent1 = self.agent = BeliefStateAgent(self.p_reward, self.p_switch, task=self.task_type)
            agents = [agent0, agent1]
            if self.config["agent"].lower() == "switchingagent":
                self.agent = SwitchingAgent(self.transition_matrix, agents, task=self.task_type)
            elif self.config["agent"].lower() == "blockswitchingagent":
                self.agent = BlockSwitchingAgent(self.transition_matrix, agents, task=self.task_type)
        else:
            raise NotImplementedError

    def run(self) -> None:
        """Run an agent through an experiment with parameters set by block_params."""
        for ep in range(self.environment.get_num_total_trials() + 1):
            self.environment.step()
            if self.environment.done:
                self.done = True
                break
            if self.task_type == BlockStimulusTask:
                stimuli = self.environment.get_current_stimulus()
                agent_action = self.agent.sample_action(stimuli)
                correct_action = self.environment.get_current_rewarded_action()
                reward = (agent_action == correct_action)
                if self.environment.current_trial_idx == 0 and type(self.agent) == BlockSwitchingAgent:
                    self.agent.update(agent_action, reward, stimuli, block_switch=True)
                else:
                    self.agent.update(agent_action, reward, stimuli)
            elif self.task_type == BlockTask:
                agent_action = self.agent.sample_action()
                correct_action = self.environment.get_current_rewarded_action()
                reward = (agent_action == correct_action)
                self.agent.update(agent_action, reward)
            else:
                raise NotImplementedError
