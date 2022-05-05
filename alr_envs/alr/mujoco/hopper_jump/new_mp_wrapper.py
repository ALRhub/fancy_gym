from alr_envs.mp.episodic_wrapper import EpisodicWrapper
from typing import Union, Tuple
import numpy as np


class NewMPWrapper(EpisodicWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[3:6].copy()

    def set_active_obs(self):
        return np.hstack([
            [False] * (5 + int(not self.env.exclude_current_positions_from_observation)),  # position
            [False] * 6,  # velocity
            [True]
        ])


class NewHighCtxtMPWrapper(NewMPWrapper):
    def set_active_obs(self):
        return np.hstack([
            [True] * (5 + int(not self.env.exclude_current_positions_from_observation)),  # position
            [False] * 6,  # velocity
            [False]
        ])