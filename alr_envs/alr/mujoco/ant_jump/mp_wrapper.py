from typing import Tuple, Union

import numpy as np

from mp_env_api import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        return np.hstack([
            [False] * 111, # ant has 111 dimensional observation space !!
            [True] # goal height
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.sim.data.qpos[7:15].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[6:14].copy()

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
