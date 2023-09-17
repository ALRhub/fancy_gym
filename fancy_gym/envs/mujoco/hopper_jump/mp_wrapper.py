from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {},
        'DMP': {},
        'ProDMP': {},
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * (2 + int(not self.exclude_current_positions_from_observation)),  # position
            [True] * 3,  # set to true if randomize initial pos
            [False] * 6,  # velocity
            [False] * 3,  # goal distance
            [True]  # goal
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[3:6].copy()
