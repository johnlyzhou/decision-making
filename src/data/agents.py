import abc
import copy
import random
from typing import Union, Type, List

import numpy as np
from numpy import ndarray

from src.data.environments import STIMULI_FREQS, ACTIONS, BOUNDARY_FREQS
from src.data.environments import SwitchingStimulusTask, DynamicForagingTask, EnvironmentInterface
from src.utils import validate_transition_matrix


class AgentInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.__action_history = []

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'action_history') and
                callable(subclass.action_history) and
                hasattr(subclass, 'sample_action') and
                callable(subclass.sample_action) and
                hasattr(subclass, 'update') and
                callable(subclass.update) or
                NotImplemented)

    @property
    def action_history(self):
        return self.__action_history

    @abc.abstractmethod
    def sample_action(self, **kwargs) -> int:
        """In the current trial, sample the agent's choice of action."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, action: int, reward: Union[int, bool], **kwargs) -> None:
        """Update agent's internal parameters based on the current trial's outcome."""
        raise NotImplementedError


class UnknownAgent(AgentInterface):
    """Dummy class to allow real data to work with agent-based functions."""
    def __init__(self, action_history: List[int]) -> None:
        super().__init__()
        self.__action_history = action_history

    def sample_action(self) -> None:
        raise NotImplementedError

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        raise NotImplementedError


class QLearningAgent(AgentInterface):
    """A model-free agent that uses Q-learning to update state-action values in response to received rewards."""
    def __init__(self, learning_rate: float, epsilon: float, task: Type[EnvironmentInterface]) -> None:
        """
        :param learning_rate: weight on the reward prediction error update.
        :param epsilon: probability of the agent taking a random action.
        :param task: task the agent is doing.
        """
        super().__init__()
        self.task = task
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        if task is SwitchingStimulusTask:
            self.action_values = [np.ones(len(ACTIONS)) / len(ACTIONS) for _ in range(len(STIMULI_FREQS))]
        elif task is DynamicForagingTask:
            self.action_values = np.ones(len(ACTIONS)) / len(ACTIONS)

        self.__action_value_history = []

    @property
    def action_value_history(self):
        return self.__action_value_history

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action for the agent to take in the current trial according to an epsilon-greedy policy.
        :param stimulus_idx: index of stimulus, referencing the imported STIMULI_FREQS list.
        :return: index of the selected action.
        """
        explore = np.random.random() < self.epsilon
        if explore:
            action = np.random.randint(0, high=len(ACTIONS))
        else:
            if self.task is SwitchingStimulusTask:
                action = np.argmax(self.action_values[stimulus_idx])
            elif self.task is DynamicForagingTask:
                action = np.argmax(self.action_values)
            else:
                raise NotImplementedError

        self.__action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        """
        Update state-action values based on the trial's outcome.
        :param stimulus_idx: index of stimulus, references the imported STIMULI_FREQS list.
        :param action: representing index of the selected action.
        :param reward: boolean or int indicating whether reward was received for the trial.
        """
        if self.task is SwitchingStimulusTask:
            self.action_values[stimulus_idx][action] += self.learning_rate * (
                    reward - self.action_values[stimulus_idx][action]
            )
            self.action_values[stimulus_idx] /= sum(self.action_values[stimulus_idx])
            self.__action_value_history.append(copy.deepcopy(self.action_values))
        elif self.task is DynamicForagingTask:
            self.action_values[action] += self.learning_rate * (reward - self.action_values[action])
            self.__action_value_history.append(self.action_values.copy())
        else:
            raise NotImplementedError


class InferenceAgent(AgentInterface):
    """A model-based agent that tracks the action side that produces a reward."""
    def __init__(self, p_reward: float, p_switch: float) -> None:
        """
        :param p_reward: agent's belief of the probability it will receive a reward if it takes the correct action.
        :param p_switch: agent's belief of the probability that the side has switched.
        """
        super().__init__()
        self.__p_reward = p_reward
        self.__p_switch = p_switch
        self.__side_beliefs = np.array([0.5, 0.5])
        self.__side_belief_history = []

    @property
    def p_reward(self):
        return self.__p_reward

    @property
    def p_switch(self):
        return self.__p_switch

    @property
    def side_belief_history(self):
        return self.__side_belief_history

    def sample_action(self) -> int:
        if self.__side_beliefs[0] > 0.5:
            action = ACTIONS["LEFT"]
        elif self.__side_beliefs[1] > 0.5:
            action = ACTIONS["RIGHT"]
        else:
            if np.random.random() < 0.5:
                action = ACTIONS["LEFT"]
            else:
                action = ACTIONS["RIGHT"]
        self.__action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        pr_l = 0
        pr_r = 0
        for side_idx in range(len(self.__side_beliefs)):
            prior = self.__side_beliefs[side_idx]

            if action == side_idx:
                if reward:
                    expected_reward_prob = self.__p_reward
                else:
                    expected_reward_prob = 1 - self.__p_reward
            else:
                if reward:
                    expected_reward_prob = 1 - self.__p_reward
                else:
                    expected_reward_prob = self.__p_reward

            if side_idx == 0:
                l_trans_probs = 1 - self.__p_switch
                r_trans_probs = self.__p_switch
            else:
                l_trans_probs = self.__p_switch
                r_trans_probs = 1 - self.__p_switch
            pr_l += prior * expected_reward_prob * l_trans_probs
            pr_r += prior * expected_reward_prob * r_trans_probs
        self.__side_beliefs = np.array([pr_l, pr_r]) / (pr_l + pr_r)
        self.__side_belief_history.append(self.__side_beliefs)


class BeliefStateAgent(AgentInterface):
    """A model-based agent that tracks the location of the boundary separating left and right stimuli."""
    def __init__(self, p_reward: float, p_switch: float) -> None:
        """
        :param p_reward: what the agent thinks is the probability of receiving a reward for the correct action.
        :param p_switch: what the agent thinks is the probability that the boundary will switch on any given trial.
        """
        super().__init__()
        self.low_boundary_idx, self.high_boundary_idx = 0, 1

        self.reward_probability = p_reward
        self.switch_probability = p_switch
        self.boundary_beliefs = np.array([0.5, 0.5])
        self.__boundary_belief_history = []

        self.reward_distribution = self._initialize_reward_distribution()
        self.reward = None
        self.perceived_stimulus = None
        self.choice = None

    @property
    def boundary_belief_history(self):
        return self.__boundary_belief_history

    def _initialize_reward_distribution(self) -> ndarray:
        """Initialize the belief distribution over rewards given an action, presented stimulus, and boundary belief."""
        left = ACTIONS['LEFT']
        right = ACTIONS['RIGHT']
        num_actions = len(ACTIONS)
        num_stimuli = len(STIMULI_FREQS)
        num_bounds = len(BOUNDARY_FREQS)

        reward_distribution = np.ones((num_actions, num_stimuli, num_bounds))
        for stimulus_idx in range(num_stimuli):
            for boundary_idx in range(num_bounds):
                if STIMULI_FREQS[stimulus_idx] < BOUNDARY_FREQS[boundary_idx]:
                    reward_distribution[left, stimulus_idx, boundary_idx] = self.reward_probability
                    reward_distribution[right, stimulus_idx, boundary_idx] = 1 - self.reward_probability
                else:
                    reward_distribution[left, stimulus_idx, boundary_idx] = 1 - self.reward_probability
                    reward_distribution[right, stimulus_idx, boundary_idx] = self.reward_probability
        return reward_distribution

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references the imported STIMULI_FREQS list.
        :return: int representing index of the selected action.
        """
        stimulus = STIMULI_FREQS[stimulus_idx]
        self.perceived_stimulus = stimulus + np.random.normal(scale=1.0)
        boundary_idx = np.argmax(self.boundary_beliefs)
        boundary = BOUNDARY_FREQS[boundary_idx]

        if self.perceived_stimulus < boundary:
            action = ACTIONS['LEFT']
        else:
            action = ACTIONS['RIGHT']

        self.__action_history.append(action)
        self.choice = action
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        """
        Recursively update beliefs over boundary location based on last trial's outcome.
        :param stimulus_idx: int indicating index of stimulus, references the imported STIMULI_FREQS list.
        :param action: int representing index of the selected action.
        :param reward: boolean or int in {0, 1} indicating whether reward was received for the trial.
        """
        low_boundary_idx = 0
        high_boundary_idx = 1

        new_boundary_beliefs = [0, 0]

        for current_boundary_belief in [low_boundary_idx, high_boundary_idx]:
            for previous_boundary_belief in [low_boundary_idx, high_boundary_idx]:
                prior = self.boundary_beliefs[previous_boundary_belief]
                expected_reward_prob = self.reward_distribution[action, stimulus_idx, previous_boundary_belief]
                if reward:
                    reward_prob = expected_reward_prob
                else:
                    reward_prob = 1 - expected_reward_prob

                if current_boundary_belief != previous_boundary_belief:
                    trans_probs = self.switch_probability
                else:
                    trans_probs = 1 - self.switch_probability
                new_boundary_beliefs[current_boundary_belief] += prior * reward_prob * trans_probs
        new_boundary_beliefs /= sum(new_boundary_beliefs)
        self.boundary_beliefs = new_boundary_beliefs
        self.__boundary_belief_history.append(new_boundary_beliefs)


class SwitchingAgent(AgentInterface):
    """Agent that switches strategies according to a transition matrix."""
    def __init__(self, transition_matrix: ndarray, agents: list[Type[AgentInterface]]) -> None:
        """
        :param transition_matrix: matrix where each entry is the probability of transitioning from the agent indexed by
        the row to the agent indexed by the column.
        :param agents: different strategies (Agent Objects) between which the agent can switch.
        """
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.current_agent_idx = 0
        self.__state_history = []

    @property
    def state_history(self):
        return self.__state_history

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: index of stimulus, references the imported STIMULI_FREQS list.
        :return: index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx)
        self.__action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        """
        Update every agent for each trial.
        """
        self.__state_history.append(self.current_agent_idx)
        for agent in self.agents:
            agent.update(action, reward, stimulus_idx=stimulus_idx)
        transition_probs = list(self.transition_matrix[self.current_agent_idx, :])
        self.current_agent_idx = random.choices(range(len(self.agents)), transition_probs)[0]


class BlockSwitchingAgent(AgentInterface):
    """Agent that switches strategies according to a transition matrix from block to block."""
    def __init__(self, transition_matrix: ndarray, agents: list[Type[AgentInterface]]) -> None:
        """
        :param transition_matrix: matrix where each entry is the probability of transitioning from the agent indexed by
        the row to the agent indexed by the column.
        :param agents: different strategies (Agent Objects) between which the agent can switch.
        """
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        self.current_agent_idx = 0
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.__action_history = []
        self.state_history = []

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references the imported STIMULI_FREQS list.
        :return: int representing index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx=stimulus_idx)
        self.__action_history.append(action)
        return action

    def update(self, action: int,
               reward: Union[int, bool],
               stimulus_idx: int = None,
               block_switch: bool = False) -> None:
        """
        All strategies are updated for each trial, but transitions between strategies only happen between blocks.
        :param stimulus_idx: int indicating index of stimulus, references the imported STIMULI_FREQS list.
        :param action: int representing index of the selected action.
        :param reward: boolean or int indicating whether reward was received for the trial (takes value 0 or 1).
        :param block_switch: boolean indicating whether or not there is a block transition.
        """
        for agent in self.agents:
            agent.update(action, reward, stimulus_idx=stimulus_idx)
        if block_switch:
            transition_probs = list(self.transition_matrix[self.current_agent_idx, :])
            self.current_agent_idx = random.choices(range(len(self.agents)), transition_probs)[0]
            self.state_history.append(self.current_agent_idx)


class RecurrentBlockSwitchingAgent(AgentInterface):
    """Agent that switches strategies (Q learning or belief state) according to a transition matrix and parameters
    of those strategies according to a continuous dynamics function."""
    def __init__(self, transition_matrix: ndarray, agents: list[Type[AgentInterface]]) -> None:
        """
        :param transition_matrix: matrix where each entry is the probability of transitioning from the agent indexed by
        the row to the agent indexed by the column.
        :param agents: different strategies (Agent Objects) between which the agent can switch.
        """
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        raise NotImplementedError

    def sample_action(self, stimulus_idx: int = None) -> int:
        raise NotImplementedError

    def update(self, action, reward, stimulus_idx=None, block_switch=False) -> None:
        raise NotImplementedError
