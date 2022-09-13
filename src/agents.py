import numpy as np


class Agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.policy = {}

    def sample_action(self):
        pass

    def step(self):
        pass

    def observe(self):
        pass

    def update(self, prev_action, reward):
        pass


class QLearningAgent(Agent):
    def __init__(self, num_actions, learning_rate, epsilon):
        super().__init__(num_actions)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_values = np.random.dirichlet(np.ones(num_actions))

    def sample_action(self):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, high=self.num_actions)
        else:
            action = np.argmax(self.action_values)
        return action

    def update(self, prev_action, reward):
        self.action_values[prev_action] += self.learning_rate * (reward - self.action_values[prev_action])


class ModelBasedAgent(Agent):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.probs = np.random.dirichlet(np.ones(num_actions))
