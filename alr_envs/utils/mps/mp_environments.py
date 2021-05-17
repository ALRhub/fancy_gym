from abc import abstractmethod
from typing import Union

import gym
import numpy as np


class MPEnv(gym.Env):

    @property
    @abstractmethod
    def active_obs(self):
        """Returns boolean value for each observation entry
        whether the observation is returned by the DMP for the contextual case or not.
        This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def start_pos(self) -> Union[float, int, np.ndarray]:
        """
        Returns the current position of the joints
        """
        raise NotImplementedError()

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray]:
        """
        Returns the current final position of the joints for the MP.
        By default this returns the starting position.
        """
        return self.start_pos
