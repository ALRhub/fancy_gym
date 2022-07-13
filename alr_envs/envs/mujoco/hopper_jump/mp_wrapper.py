from typing import Union, Tuple

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * (2 + int(not self.exclude_current_positions_from_observation)),  # position
            [True] * 3,  # set to true if randomize initial pos
            [False] * 6,  # velocity
            [True] * 3,  # goal distance
            [True]  # goal
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[3:6].copy()
