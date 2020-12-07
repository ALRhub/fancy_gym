import gym
import numpy as np

from alr_envs.stochastic_search.functions.f_base import BaseObjective


class StochasticSearchEnv(gym.Env):

    def __init__(self, cost_f: BaseObjective):
        self.cost_f = cost_f

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cost_f.dim,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=(), high=(), shape=(), dtype=np.float64)

    def step(self, action):
        return np.zeros(self.observation_space.shape), np.squeeze(-self.cost_f(action)), True, {}

    def reset(self):
        return np.zeros(self.observation_space.shape)

    def render(self, mode='human'):
        pass
