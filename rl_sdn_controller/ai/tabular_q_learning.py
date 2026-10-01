"""Small tabular Q-learning baseline for the SDN routing environment."""

import random

import numpy as np


class TabularQLearningAgent:
    def __init__(self, action_dim: int, gamma: float = 0.99, learning_rate: float = 0.1):
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.q_table = {}

    @staticmethod
    def _state(observation):
        # Aggregate five normalized features per link into a compact table key.
        links = np.asarray(observation, dtype=np.float32).reshape(-1, 5)
        means = np.clip(links[:, :4].mean(axis=0), 0.0, 1.0)
        bins = tuple(min(2, int(value * 3)) for value in means)
        return bins + (int(np.any(links[:, 4] < 0.5)),)

    def _values(self, state):
        return self.q_table.setdefault(state, np.zeros(self.action_dim, dtype=np.float64))

    def select_action(self, observation, evaluate: bool = False, epsilon: float = 0.0):
        if not evaluate and random.random() < epsilon:
            return random.randrange(self.action_dim)
        values = self._values(self._state(observation))
        return int(np.argmax(values))

    def update(self, observation, action: int, reward: float, next_observation, done: bool):
        values = self._values(self._state(observation))
        next_values = self._values(self._state(next_observation))
        target = reward + (0.0 if done else self.gamma * float(np.max(next_values)))
        values[action] += self.learning_rate * (target - values[action])
