from typing import Union

import numpy as np

from mp_env_api.envs.mp_env_wrapper import MPEnvWrapper


class ViaPointReacherMPWrapper(MPEnvWrapper):
    @property
    def active_obs(self):
        return np.hstack([
            [self.env.random_start] * self.env.n_links,  # cos
            [self.env.random_start] * self.env.n_links,  # sin
            [self.env.random_start] * self.env.n_links,  # velocity
            [self.env.initial_via_target is None] * 2,  # x-y coordinates of via point distance
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def start_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.start_pos

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
