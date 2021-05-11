from abc import abstractmethod
from typing import Union, Tuple

import numpy as np
from gym import Env

class PositionalEnv(Env):
    """A position and velocity based environment. It functions just as any regular OpenAI Gym
    environment but it provides position, velocity and acceleration information. This usually means that the
    corresponding information from the agent is forwarded via the properties.
    PD-Controller based policies require this environment to calculate the state dependent actions for example.
    """

    @property
    @abstractmethod
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        raise NotImplementedError
