from typing import Union, Tuple

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self):
        return np.concatenate([
            [False] * self.env.n_links,  # cos
            [False] * self.env.n_links,  # sin
            [True] * 2,  # goal position
            [False] * self.env.n_links,  # angular velocity
            [False] * 3,  # goal distance
            # [False],  # step
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.data.qpos.flat[:self.env.n_links]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.data.qvel.flat[:self.env.n_links]
