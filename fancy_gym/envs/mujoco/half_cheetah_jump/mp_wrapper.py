from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {},
        'DMP': {},
        'ProDMP': {},
    }

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [False] * 17,
            [True]  # goal height
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.data.qpos[3:9].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[3:9].copy()
