import random

import numpy as np

from src.data.environments import STIMULI_FREQS, ACTIONS, BOUNDARY_FREQS


def validate_transition_matrix(transition_matrix):
    if len(transition_matrix.shape) != 2:
        raise ValueError("Transition matrix should be 2D!")
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix should be symmetrical!")
    for row in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row]) != 1:
            raise ValueError("Rows of transition probabilities should sum to 1!")


class QLearningAgent:
    """A model-free agent that uses Q-learning to update state-action values in response to received rewards."""
    def __init__(self, learning_rate, epsilon):
        """
        :param learning_rate: float in range (0, inf] weighting the update of the current reward prediction error.
        :param epsilon: float in range (0, 1] indicating probability of the agent taking an exploratory action.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        # Initialize a uniform probability distribution over actions for each stimulus.
        self.stimulus_action_values = [np.ones(len(ACTIONS)) / len(ACTIONS) for _ in range(len(STIMULI_FREQS))]
        self.stimulus_action_value_history = []
        self.action_history = []

    def sample_action(self, stimulus_idx):
        """
        Choose an action according to an epsilon-greedy policy.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
        :return: int representing index of the selected action.
        """
        explore = np.random.random() < self.epsilon
        if explore:
            action = np.random.randint(0, high=len(ACTIONS))
        else:
            action = np.argmax(self.stimulus_action_values[stimulus_idx])
        self.action_history.append(action)
        return action

    def update(self, stimulus_idx, action, reward):
        """
        Update state-action value based on the observed rewards in this trial.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
        :param action: int representing index of the selected action.
        :param reward: boolean or int indicating whether reward was received for the trial (takes value 0 or 1).
        """
        self.stimulus_action_values[stimulus_idx][action] += self.learning_rate * (
                reward - self.stimulus_action_values[stimulus_idx][action]
        )
        self.stimulus_action_value_history.append(self.stimulus_action_values)


class BeliefStateAgent:
    """A model-based agent that tracks the location of the boundary separating left and right stimuli."""
    def __init__(self, p_reward, p_switch):
        """
        :param p_reward: float in range [0, 1] indicating the agent's belief of the probability it will receive a reward
        if it takes the correct action.
        :param p_switch: float in range [0, 1] indicating the agent's belief of the probability that the boundary has
        switched.
        """
        super().__init__()
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

    def initialize_reward_distribution(self):
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

    def sample_action(self, stimulus_idx):
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
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

    def update(self, stimulus_idx, action, reward):
        """
        Recursively update beliefs over boundary location based on last trial's outcome.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
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


class SwitchingAgent:
    """Agent that switches strategies according to a transition matrix."""
    def __init__(self, transition_matrix, agents):
        """
        :param transition_matrix: symmetrical numpy matrix where each entry is the probability of transitioning from the
        agent indexed by the row to the agent indexed by the column. Elements in range [0, 1] and rows should sum to 1.
        :param agents: list of Agent objects.
        """
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        self.current_agent_idx = 0
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.action_history = []
        self.state_history = []

    def sample_action(self, stimulus_idx):
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
        :return: int representing index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx)
        self.action_history.append(action)
        return action

    def update(self, stimulus_idx, action, reward):
        self.state_history.append(self.current_agent_idx)
        """All strategies are updated for each trial."""
        for agent in self.agents:
            agent.update(stimulus_idx, action, reward)
        transition_probs = list(self.transition_matrix[self.current_agent_idx, :])
        self.current_agent_idx = random.choices(range(len(self.agents)), transition_probs)[0]


class BlockSwitchingAgent:
    """Agent that switches strategies according to a transition matrix from block to block."""
    def __init__(self, transition_matrix, agents):
        """
        :param transition_matrix: symmetrical numpy matrix where each entry is the probability of transitioning from the
        agent indexed by the row to the agent indexed by the column.
        :param agents: list of Agent objects.
        """
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        self.agents = agents
        self.current_agent_idx = 0
        if self.transition_matrix.shape[0] != len(self.agents) or self.transition_matrix.shape[1] != len(self.agents):
            raise ValueError("Transition matrix shape should match number of agents!")
        self.action_history = []
        self.state_history = []

    def sample_action(self, stimulus_idx):
        """
        Sample an action according to a noisy stimulus perception and belief over current boundary location.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
        :return: int representing index of the selected action.
        """
        action = self.agents[self.current_agent_idx].sample_action(stimulus_idx)
        self.action_history.append(action)
        return action

    def update(self, stimulus_idx, action, reward, block_switch=False):
        """
        All strategies are updated for each trial, but transitions between strategies only happen between blocks.
        :param stimulus_idx: int indicating index of stimulus, references imported STIMULI_FREQS.
        :param action: int representing index of the selected action.
        :param reward: boolean or int indicating whether reward was received for the trial (takes value 0 or 1).
        :param block_switch: boolean indicating whether or not there is a block transition.
        """
        for agent in self.agents:
            agent.update(stimulus_idx, action, reward)
        if block_switch:
            transition_probs = list(self.transition_matrix[self.current_agent_idx, :])
            self.current_agent_idx = random.choices(range(len(self.agents)), transition_probs)[0]
            self.state_history.append(self.current_agent_idx)


class RecurrentSwitchingAgent:
    """Agent that switches strategies (Q learning or belief state) according to a transition matrix and parameters
    of those strategies according to a continuous dynamics function."""
    def __init__(self, transition_matrix):
        validate_transition_matrix(transition_matrix)
        self.transition_matrix = transition_matrix
        pass
