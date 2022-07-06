from typing import Union, Tuple

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self):
        return np.hstack([
            [False] * 111,  # ant has 111 dimensional observation space !!
            [True]  # goal height
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.sim.data.qpos[7:15].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[6:14].copy()
