from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    mp_config = {
        'ProMP': {},
        'DMP': {
            'phase_generator_kwargs': {
                'alpha_phase': 2,
            },
        },
        'ProDMP': {},
    }

    @property
    def context_mask(self):
        return np.concatenate([[False] * self.n_links,  # cos
                               [False] * self.n_links,  # sin
                               [True] * 2,  # goal position
                               [False] * self.n_links,  # angular velocity
                               [False] * 3,  # goal distance
                               # [False],  # step
                               ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos.flat[:self.n_links]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel.flat[:self.n_links]
