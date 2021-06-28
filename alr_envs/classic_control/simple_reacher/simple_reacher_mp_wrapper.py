from typing import Union

import numpy as np

from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class SimpleReacherMPWrapper(MPEnvWrapper):
    @property
    def active_obs(self):
        return np.hstack([
            [self.env.random_start] * self.env.n_links,  # cos
            [self.env.random_start] * self.env.n_links,  # sin
            [self.env.random_start] * self.env.n_links,  # velocity
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def start_pos(self):
        return self.env.start_pos

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
