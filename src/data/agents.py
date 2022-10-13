import abc
import copy
import random
from typing import Union, Type

import numpy as np
from numpy import ndarray

from src.data.environments import STIMULI_FREQS, ACTIONS, BOUNDARY_FREQS, BlockStimulusTask, BlockTask, \
    EnvironmentInterface
from src.utils import validate_transition_matrix


class AgentInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample_action') and
                callable(subclass.sample_action) and
                hasattr(subclass, 'update') and
                callable(subclass.update) or
                NotImplemented)

    @abc.abstractmethod
    def sample_action(self, **kwargs) -> int:
        """Sample an action for the agent to take in the current trial."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, action: int, reward: Union[int, bool], **kwargs) -> None:
        """Update agent's internal parameters based on the trial's outcome."""
        raise NotImplementedError


class RealAgent(AgentInterface):
    """A model-free agent that uses Q-learning to update state-action values in response to received rewards."""
    def __init__(self) -> None:
        super().__init__()
        self.action_history = []

    def sample_action(self) -> None:
        raise NotImplementedError

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        raise NotImplementedError


class QLearningAgent(AgentInterface):
    """A model-free agent that uses Q-learning to update state-action values in response to received rewards."""
    def __init__(self, learning_rate: float, epsilon: float, task: Type[EnvironmentInterface]) -> None:
        """
        :param learning_rate: weight on the update of the current reward prediction error.
        :param epsilon: probability of the agent taking an exploratory action.
        :param task: type of task agent is doing.
        """
        super().__init__()
        self.task = task
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_history = []

        if task is BlockStimulusTask:
            self.stimulus_action_values = [np.ones(len(ACTIONS)) / len(ACTIONS) for _ in range(len(STIMULI_FREQS))]
            self.stimulus_action_value_history = []
        elif task is BlockTask:
            self.action_values = np.ones(len(ACTIONS)) / len(ACTIONS)
            self.action_value_history = []

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
            if self.task is BlockStimulusTask:
                action = np.argmax(self.stimulus_action_values[stimulus_idx])
            elif self.task is BlockTask:
                action = np.argmax(self.action_values)
            else:
                raise NotImplementedError

        self.action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        """
        Update state-action values based on the trial's outcome.
        :param stimulus_idx: index of stimulus, references the imported STIMULI_FREQS list.
        :param action: representing index of the selected action.
        :param reward: boolean or int indicating whether reward was received for the trial.
        """
        if self.task is BlockStimulusTask:
            self.stimulus_action_values[stimulus_idx][action] += self.learning_rate * (
                    reward - self.stimulus_action_values[stimulus_idx][action]
            )
            self.stimulus_action_values[stimulus_idx] /= sum(self.stimulus_action_values[stimulus_idx])
            self.stimulus_action_value_history.append(copy.deepcopy(self.stimulus_action_values))
        elif self.task is BlockTask:
            self.action_values[action] += self.learning_rate * (reward - self.action_values[action])
            # self.action_values /= np.sum(self.action_values)
            self.action_value_history.append(self.action_values.copy())
        else:
            raise NotImplementedError


class InferenceAgent(AgentInterface):
    """A model-based agent that tracks the action side that produces a reward."""
    def __init__(self, p_reward: float, p_switch: float, task: Type[EnvironmentInterface]) -> None:
        """
        :param p_reward: agent's belief of the probability it will receive a reward if it takes the correct action.
        :param p_switch: agent's belief of the probability that the side has switched.
        """
        super().__init__()
        self.task = task
        if self.task is not BlockTask:
            raise ValueError("The InferenceAgent is designed for the BlockTask only!")
        self.p_reward = p_reward
        self.p_switch = p_switch
        self.side_beliefs = np.array([0.5, 0.5])
        self.action_history = []
        self.side_belief_history = []

    def sample_action(self) -> int:
        if self.side_beliefs[0] > 0.5:
            action = ACTIONS["LEFT"]
        elif self.side_beliefs[1] > 0.5:
            action = ACTIONS["RIGHT"]
        else:
            if np.random.random() < 0.5:
                action = ACTIONS["LEFT"]
            else:
                action = ACTIONS["RIGHT"]
        self.action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        pr_l = 0
        pr_r = 0
        for side_idx in range(len(self.side_beliefs)):
            prior = self.side_beliefs[side_idx]

            if action == side_idx:
                if reward:
                    expected_reward_prob = self.p_reward
                else:
                    expected_reward_prob = 1 - self.p_reward
            else:
                if reward:
                    expected_reward_prob = 1 - self.p_reward
                else:
                    expected_reward_prob = self.p_reward

            if side_idx == 0:
                l_trans_probs = 1 - self.p_switch
                r_trans_probs = self.p_switch
            else:
                l_trans_probs = self.p_switch
                r_trans_probs = 1 - self.p_switch
            pr_l += prior * expected_reward_prob * l_trans_probs
            pr_r += prior * expected_reward_prob * r_trans_probs
        self.side_beliefs = np.array([pr_l, pr_r]) / (pr_l + pr_r)
        self.side_belief_history.append(self.side_beliefs)


class BeliefStateAgent(AgentInterface):
    """A model-based agent that tracks the location of the boundary separating left and right stimuli."""
    def __init__(self, p_reward: float, p_switch: float, task: Type[EnvironmentInterface]) -> None:
        """
        :param p_reward: agent's belief of the probability it will receive a reward if it takes the correct action.
        :param p_switch: agent's belief of the probability that the boundary has switched.
        """
        super().__init__()
        if task is not BlockStimulusTask:
            raise ValueError("The BeliefStateAgent is designed for the BlockStimulusTask only!")
        self.low_boundary_idx = 0
        self.high_boundary_idx = 1
        self.boundary_beliefs = np.array([0.5, 0.5])
        self.reward_probability = p_reward
        self.reward_distribution = None
        self.initialize_reward_distribution()
        self.switch_probability = p_switch
        self.reward = None
        self.perceived_stimulus = None
        self.choice = None
        self.action_history = []
        self.boundary_belief_history = []

    def initialize_reward_distribution(self) -> None:
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
        self.reward_distribution = reward_distribution

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

        self.action_history.append(action)
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
        self.boundary_belief_history.append(new_boundary_beliefs)


class SwitchingAgent(AgentInterface):
    """Agent that switches strategies according to a transition matrix."""
    def __init__(self, transition_matrix: ndarray,
                 agents: list,
                 task: Type[EnvironmentInterface]) -> None:
        """
        :param transition_matrix: symmetrical numpy matrix where each entry is the probability of transitioning from the
        agent indexed by the row to the agent indexed by the column. Elements in range [0, 1] and rows should sum to 1.
        :param agents: list of Agent objects.
        """
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        self.current_agent_idx = 0
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.action_history = []
        self.state_history = []

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: index of stimulus, references the imported STIMULI_FREQS list.
        :return: index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx)
        self.action_history.append(action)
        return action

    def update(self, action: int, reward: Union[int, bool], stimulus_idx: int = None) -> None:
        self.state_history.append(self.current_agent_idx)
        """All strategies are updated for each trial."""
        for agent in self.agents:
            agent.update(action, reward, stimulus_idx=stimulus_idx)
        transition_probs = list(self.transition_matrix[self.current_agent_idx, :])
        self.current_agent_idx = random.choices(range(len(self.agents)), transition_probs)[0]


class BlockSwitchingAgent(AgentInterface):
    """Agent that switches strategies according to a transition matrix from block to block."""
    def __init__(self, transition_matrix: ndarray,
                 agents: list,
                 task: Type[EnvironmentInterface]) -> None:
        """
        :param transition_matrix: symmetrical matrix where each entry is the probability of transitioning from the
        strategy indexed by the row to the strategy indexed by the column.
        :param agents: different strategies that the agent can switch between.
        """
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        self.current_agent_idx = 0
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.action_history = []
        self.state_history = []

    def sample_action(self, stimulus_idx: int = None) -> int:
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references the imported STIMULI_FREQS list.
        :return: int representing index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx=stimulus_idx)
        self.action_history.append(action)
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
    def __init__(self, transition_matrix: ndarray, task: Type[EnvironmentInterface]) -> None:
        super().__init__()
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        raise NotImplementedError

    def sample_action(self, stimulus_idx: int = None) -> int:
        raise NotImplementedError

    def update(self, action, reward, stimulus_idx=None, block_switch=False) -> None:
        raise NotImplementedError
