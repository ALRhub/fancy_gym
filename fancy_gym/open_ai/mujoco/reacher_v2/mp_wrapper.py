from typing import Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qvel[:2]

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qpos[:2]

    @property
    def context_mask(self):
        return np.concatenate([
            [False] * 2,  # cos of two links
            [False] * 2,  # sin of two links
            [True] * 2,  # goal position
            [False] * 2,  # angular velocity
            [False] * 3,  # goal distance
        ])
