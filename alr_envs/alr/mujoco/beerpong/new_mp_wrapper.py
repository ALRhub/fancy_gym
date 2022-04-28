from mp_wrapper import BaseMPWrapper
from typing import Union, Tuple
import numpy as np


class MPWrapper(BaseMPWrapper):

    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos[0:7].copy()

    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[0:7].copy()

    def set_active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [True] * 2,  # xy position of cup
            [False]  # env steps
        ])
