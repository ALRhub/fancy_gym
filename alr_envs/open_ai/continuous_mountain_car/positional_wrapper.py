from typing import Union
import numpy as np
from mp_env_api.env_wrappers.positional_env_wrapper import PositionalEnvWrapper


class PositionalWrapper(PositionalEnvWrapper):
    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        return np.array([self.state[1]])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return np.array([self.state[0]])