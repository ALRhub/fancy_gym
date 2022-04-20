from typing import Tuple, Union

import numpy as np

from mp_env_api import MPEnvWrapper


class MPWrapper(MPEnvWrapper):
    @property
    def active_obs(self):
        return np.hstack([
            [False] * (5 + int(not self.exclude_current_positions_from_observation)),  # position
            [False] * 6,  # velocity
            [True]
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.sim.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[3:6].copy()

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt


class HighCtxtMPWrapper(MPWrapper):
    @property
    def active_obs(self):
        return np.hstack([
            [True] * (5 + int(not self.exclude_current_positions_from_observation)),  # position
            [False] * 6,  # velocity
            [False]
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.sim.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[3:6].copy()

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
