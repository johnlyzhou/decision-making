import numpy as np

from src.switch_2afc_task import STIMULI_FREQS, ACTIONS, BOUNDARY_IDX, BOUNDARY_FREQS


class QLearningAgent:
    """A model-free agent that uses Q-learning to update state-action values in response to received rewards."""
    def __init__(self, learning_rate, epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        # Initialize a uniform probability distribution over actions for each stimulus.
        self.stimulus_action_values = [np.ones(len(ACTIONS)) / len(ACTIONS) for _ in range(len(STIMULI_FREQS))]
        self.stimulus_action_value_history = []
        self.action_history = []

    def sample_action(self, stimulus_idx):
        """Choose an action according to an epsilon-greedy policy."""
        explore = np.random.random() < self.epsilon
        if explore:
            action = np.random.randint(0, high=len(ACTIONS))
        else:
            action = np.argmax(self.stimulus_action_values[stimulus_idx])
        self.action_history.append(action)
        return action

    def update(self, stimulus_idx, action, reward):
        """Update state-action value based on the observed rewards in this trial."""
        self.stimulus_action_values[stimulus_idx][action] += self.learning_rate * (
                reward - self.stimulus_action_values[stimulus_idx][action]
        )
        self.stimulus_action_value_history.append(self.stimulus_action_values)


class BeliefStateAgent:
    """A model-based agent that tracks the location of the boundary separating left and right stimuli."""
    def __init__(self, p_reward, p_switch):
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
        """Sample an action according to a noisy stimulus perception and belief over current boundary location."""
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
        """Recursively update beliefs over boundary location based on last trial's outcome."""
        # Use to index into reward distribution and new boundary beliefs.
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
