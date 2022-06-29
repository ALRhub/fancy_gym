from typing import Tuple, Union

import numpy as np

from mp_env_api import MPEnvWrapper

from alr_envs.mp.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    def context_mask(self):
        return np.hstack([
            [self.env.random_start] * self.env.n_links,  # cos
            [self.env.random_start] * self.env.n_links,  # sin
            [self.env.random_start] * self.env.n_links,  # velocity
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
