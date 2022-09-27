import numpy as np

from src.environments import STIMULI, ACTIONS, LOW_IDX, HIGH_IDX, LEFT, RIGHT, BOUNDARIES


class Agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def sample_action(self, stimulus_idx):
        pass

    def step(self):
        pass

    def observe(self):
        pass

    def update(self, stimulus_idx, action, reward):
        pass


class QLearningAgent(Agent):
    def __init__(self, num_stimuli, num_actions, learning_rate, epsilon):
        super().__init__(num_actions)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.stimulus_action_values = [np.ones(num_actions) / num_actions for _ in range(num_stimuli)]
        self.stimulus_action_value_history = []
        self.action_history = []

    def sample_action(self, stimulus_idx):
        explore = np.random.random() < self.epsilon
        if explore:
            action = np.random.randint(0, high=self.num_actions)
        else:
            action = np.argmax(self.stimulus_action_values[stimulus_idx])
        self.action_history.append(action)
        return action

    def update(self, stimulus_idx, action, reward):
        self.stimulus_action_values[stimulus_idx][action] += self.learning_rate * (
                reward - self.stimulus_action_values[stimulus_idx][action]
        )
        self.stimulus_action_value_history.append(self.stimulus_action_values)


class BeliefStateAgent(Agent):
    def __init__(self, num_actions, p_reward, p_switch):
        super().__init__(num_actions)
        self.boundary_beliefs = np.array([0.5, 0.5])
        self.p_reward = p_reward
        self.reward_distribution = self.initialize_rew_dist(ACTIONS, STIMULI, BOUNDARIES)
        self.p_switch = p_switch
        self.reward = None
        self.perceived_stimulus = None
        self.choice = None

    def initialize_rew_dist(self, actions, stimuli, boundaries):
        num_actions = len(actions)
        num_stimuli = len(stimuli)
        num_bounds = len(boundaries)
        rew_dist = np.ones((num_actions, num_stimuli, num_bounds))
        for stim_idx in range(num_stimuli):
            for bound_idx in range(num_bounds):
                if STIMULI[stim_idx] < BOUNDARIES[bound_idx]:
                    rew_dist[LEFT, stim_idx, bound_idx] = self.p_reward
                    rew_dist[RIGHT, stim_idx, bound_idx] = 1 - self.p_reward
                else:
                    rew_dist[LEFT, stim_idx, bound_idx] = 1 - self.p_reward
                    rew_dist[RIGHT, stim_idx, bound_idx] = self.p_reward
        return rew_dist

    def sample_action(self, stimulus_idx):
        stimulus = STIMULI[stimulus_idx]
        self.perceived_stimulus = stimulus + np.random.normal(scale=1.0)
        boundary_idx = np.argmax(self.boundary_beliefs)
        boundary = BOUNDARIES[boundary_idx]

        if self.perceived_stimulus < boundary:
            self.choice = ACTIONS[0]
        else:
            self.choice = ACTIONS[1]

        return self.choice

    def update(self, stimulus_idx, action, reward):
        current_bound_beliefs = [0, 0]
        for curr_idx in [LOW_IDX, HIGH_IDX]:
            for prev_idx in [LOW_IDX, HIGH_IDX]:
                prior = self.boundary_beliefs[prev_idx]
                rew_prob = self.reward_distribution[action, stimulus_idx, prev_idx]
                if curr_idx != prev_idx:
                    trans_probs = self.p_switch
                else:
                    trans_probs = 1 - self.p_switch
                current_bound_beliefs[curr_idx] += prior * rew_prob * trans_probs
        current_bound_beliefs /= sum(current_bound_beliefs)
        self.boundary_beliefs = current_bound_beliefs
        return current_bound_beliefs
