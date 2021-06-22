from abc import abstractmethod, ABC
from typing import Union

import gym
import numpy as np


class AlrEnv(gym.Env, ABC):

    @property
    def active_obs(self):
        """Returns boolean mask for each observation entry
        whether the observation is returned for the contextual case or not.
        This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
        """
        return np.ones(self.observation_space.shape, dtype=bool)

    @property
    @abstractmethod
    def start_pos(self) -> Union[float, int, np.ndarray]:
        """
        Returns the starting position of the joints
        """
        raise NotImplementedError()

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray]:
        """
        Returns the current final position of the joints for the MP.
        By default this returns the starting position.
        """
        return self.start_pos

    @property
    @abstractmethod
    def dt(self) -> Union[float, int]:
        """
        Returns the time between two simulated steps of the environment
        """
        raise NotImplementedError()