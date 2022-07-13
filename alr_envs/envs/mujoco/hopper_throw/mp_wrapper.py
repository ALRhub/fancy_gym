from typing import Tuple, Union

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self):
        return np.hstack([
            [False] * 17,
            [True]  # goal pos
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.data.qvel[3:6].copy()