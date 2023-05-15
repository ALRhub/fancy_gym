import gymnasium as gym
import numpy as np


class TimeAwareObservation(gym.wrappers.TimeAwareObservation):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._max_episode_steps = env.spec.max_episode_steps

    def observation(self, observation):
        """Adds to the observation with the current time step normalized with max steps.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        return np.append(observation, self.t / self._max_episode_steps)
