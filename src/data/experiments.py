import abc
from typing import Union, Any, Tuple, Type

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from src.data.agents import QLearningAgent, BeliefStateAgent, BlockSwitchingAgent, InferenceAgent, \
    UnknownAgent
from src.data.environments import SwitchingStimulusTask, DynamicForagingTask, EnvironmentInterface
from src.data.real_data import DynamicForagingData, generate_real_block_params, convert_real_actions


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
    def __init__(self, filename: str = None, task_type: Type[EnvironmentInterface] = None):
        """
        :param filename: Path to MATLAB file generated from a real experiment.
        :param task_type: Type of task.
        """
        super().__init__()
        self.__done = True
        self.task_type = task_type
        if self.task_type is DynamicForagingTask:
            self.data = DynamicForagingData(filename)
            self.blocks = generate_real_block_params(self.data.block, self.data.correct_side,
                                                     real_actions=self.data.response_side)
            self.action_history = convert_real_actions(self.data.response_side)
            self.reward_history = self.data.rewarded
            self.environment = DynamicForagingTask(self.blocks, self.reward_history)
            self.agent = UnknownAgent(self.action_history)
        else:
            raise NotImplementedError

    def run(self):
        raise NotImplementedError


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
