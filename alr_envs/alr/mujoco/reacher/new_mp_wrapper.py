from alr_envs.mp.episodic_wrapper import EpisodicWrapper
from typing import Union, Tuple
import numpy as np


class MPWrapper(EpisodicWrapper):

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos.flat[:self.env.n_links]
    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel.flat[:self.env.n_links]

    def set_active_obs(self):
        return np.concatenate([
            [False] * self.env.n_links,  # cos
            [False] * self.env.n_links,  # sin
            [True] * 2,  # goal position
            [False] * self.env.n_links,  # angular velocity
            [False] * 3,  # goal distance
            # self.get_body_com("target"),  # only return target to make problem harder
            [False],  # step
        ])