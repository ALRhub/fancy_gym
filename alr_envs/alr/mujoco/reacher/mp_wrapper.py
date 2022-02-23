from typing import Union

import numpy as np
from mp_env_api import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        return np.concatenate([
            [True] * self.n_links,  # cos
            [True] * self.n_links,  # sin
            [True] * 2,  # goal position
            [True] * self.n_links,  # angular velocity
            [True] * 2,  # goal distance
            # self.get_body_com("target"),  # only return target to make problem harder
            [False], # step
        ])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qvel.flat[:self.n_links]

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qpos.flat[:self.n_links]

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
