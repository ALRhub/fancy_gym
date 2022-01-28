from typing import Tuple, Union

import numpy as np

from mp_env_api.interface_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [True] * 2,  # xy position of cup
            [False]  # env steps
        ])

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qvel[0:7].copy()

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
