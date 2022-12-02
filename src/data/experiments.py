import abc
from typing import Union, Any, Tuple, Type

import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from src.data.agents import QLearningAgent, BeliefStateAgent, BlockSwitchingAgent, InferenceAgent, \
    UnknownAgent
from src.data.environments import SwitchingStimulusTask, DynamicForagingTask, EnvironmentInterface
from src.data.real_data import RealSessionDataset, generate_real_block_params, convert_real_actions
from src.utils import blockify, normalize_choice_block_side, truncate_blocks


class ExperimentInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.__done = False

    @property
    def done(self):
        return self.__done

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    @abc.abstractmethod
    def run(self, **kwargs) -> int:
        raise NotImplementedError


class RealExperiment(ExperimentInterface):
    def __init__(self, filename: str = None,
                 task_type: Type[EnvironmentInterface] = None,
                 remove_nans: bool = True):
        """
        :param filename: Path to MATLAB file generated from a real experiment.
        :param task_type: Type of task.
        :param remove_nans: Whether to remove trials where the animal doesn't make a choice
        """
        super().__init__()
        self.__done = True
        self.task_type = task_type

        if self.task_type is DynamicForagingTask:
            self.dataset = RealSessionDataset(filename)

            if remove_nans:
                self.valid_idxs = np.argwhere(~np.isnan(self.dataset.actions)).flatten()
            else:
                self.valid_idxs = np.arange(self.dataset.actions.size)

            self.num_trials = self.valid_idxs.size

            self.blocks = generate_real_block_params(self.dataset.blocks[self.valid_idxs],
                                                     self.dataset.correct_side[self.valid_idxs])
            self.action_history = convert_real_actions(self.dataset.actions[self.valid_idxs])
            self.reward_history = self.dataset.rewarded[self.valid_idxs]
            self.environment = DynamicForagingTask(self.blocks, self.reward_history)
            self.agent = UnknownAgent(list(self.action_history))

            self.response_wait_start_time = None
            self.first_response_time = None
            self.second_response_time = None
            self.reaction_time = None
            self.inter_reaction_delay = None
            self.states = self.dataset.states[self.valid_idxs]

            if remove_nans:
                self.response_wait_start_time = np.array([trial_state['WaitForResponse'][0]
                                                          if trial_state['WaitForResponse'].ndim == 1
                                                          else trial_state['WaitForResponse'][0][0]
                                                          for trial_state in self.states])
                self.first_response_time, self.second_response_time = self._get_response_times()
                self.reaction_time = self.first_response_time - self.response_wait_start_time
                self.inter_reaction_delay = self.second_response_time - self.first_response_time
        else:
            raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_preprocessed_blocks(self, min_len: int = 15):
        blocked_actions = blockify(self.blocks, list(self.action_history))
        normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], side=self.blocks[block_idx][0])
                              for block_idx in range(len(self.blocks))]
        truncated_actions = truncate_blocks(normalized_actions, truncate_length=min_len)
        return [choice_block for choice_block in truncated_actions if len(choice_block) >= min_len]

    def _get_response_times(self):
        first_response_time = np.zeros(self.num_trials)
        second_response_time = np.zeros(self.num_trials)
        for idx in range(self.num_trials):
            if self.reward_history[idx]:
                if self.states[idx]['CheckReward'].ndim == 1:
                    first_response_time[idx], second_response_time[idx] = self.states[idx]['CheckReward']
                else:
                    first_response_time[idx], second_response_time[idx] = self.states[idx]['CheckReward'][-1]
            else:
                if self.states[idx]['CheckPunish'].ndim == 1:
                    first_response_time[idx], second_response_time[idx] = self.states[idx]['CheckPunish']
                else:
                    first_response_time[idx], second_response_time[idx] = self.states[idx]['CheckPunish'][-1]
        return first_response_time, second_response_time


class BasicSynthExperiment(ExperimentInterface):
    def __init__(self, agent, environment):
        super().__init__()
        self.agent = agent
        self.environment = environment

    def run(self):
        for ep in range(len(self.environment) + 1):
            self.environment.step()

            if self.environment.done:
                self.__done = True
                break

            if type(self.environment) == DynamicForagingTask:
                agent_action = self.agent.sample_action()
                correct_action = self.environment.get_current_rewarded_action()
                reward = (agent_action == correct_action)
                self.agent.update(agent_action, reward, block_switch=(self.environment.current_trial_idx == 0))
            else:
                raise NotImplementedError


class SynthExperiment(ExperimentInterface):
    """Sets up a task environment and agent(s) and runs a synthetic experiment based on configuration settings."""
    def __init__(self, config: Union[str, DictConfig] = None) -> None:
        """
        :param config: Path to config file or a dictionary config.
        """
        super().__init__()
        if type(config) == str:
            self.config = OmegaConf.load(config)
        else:
            self.config = config

        self.task_type, self.blocks, self.environment = self.__init_environment()
        self.agent = self.__init_agent()

    def run(self) -> None:
        """Run an agent through an environment."""
        for ep in range(len(self.environment) + 1):
            self.environment.step()
            if self.environment.done:
                self.__done = True
                break
            if self.task_type == DynamicForagingTask:
                agent_action = self.agent.sample_action()
                correct_action = self.environment.get_current_rewarded_action()
                reward = (agent_action == correct_action)
                self.agent.update(agent_action, reward)
            elif self.task_type == SwitchingStimulusTask:
                stimuli = self.environment.get_current_stimulus()
                agent_action = self.agent.sample_action(stimuli)
                correct_action = self.environment.get_current_rewarded_action()
                reward = (agent_action == correct_action)
                if self.environment.current_trial_idx == 0 and type(self.agent) == BlockSwitchingAgent:
                    self.agent.update(agent_action, reward, stimuli, block_switch=True)
                else:
                    self.agent.update(agent_action, reward, stimuli)
            else:
                raise NotImplementedError

    def __init_environment(self) -> Tuple[Any, Tuple, Any]:
        if self.config["environment"].lower() == "switchingstimulustask":
            task_type = SwitchingStimulusTask
            blocks = self.config["blocks"]
            environment = SwitchingStimulusTask(blocks)
        elif self.config["environment"].lower() == "dynamicforagingtask":
            task_type = DynamicForagingTask
            blocks = self.config["blocks"]
            environment = DynamicForagingTask(blocks)
        else:
            raise NotImplementedError

        return task_type, blocks, environment

    def __init_agent(self) -> Any:
        if self.config["agent"].lower() == "qlearningagent":
            learning_rate = self.config["learning_rate"]
            epsilon = self.config["epsilon"]
            agent = QLearningAgent(learning_rate, epsilon, task=self.task_type)
        elif self.config["agent"].lower() == "inferenceagent":
            p_reward = self.config["p_reward"]
            p_switch = self.config["p_switch"]
            agent = InferenceAgent(p_reward, p_switch)
        elif self.config["agent"].lower() == "beliefstateagent":
            p_reward = self.config["p_reward"]
            p_switch = self.config["p_switch"]
            agent = BeliefStateAgent(p_reward, p_switch)
        else:
            raise NotImplementedError

        return agent
