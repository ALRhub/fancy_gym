from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'controller_type': 'velocity',
            },
        },
        'DMP': {
            'controller_kwargs': {
                'controller_type': 'velocity',
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 50,
            },
            'phase_generator_kwargs': {
                'alpha_phase': 2,
            },
        },
        'ProDMP': {},
    }

    @property
    def context_mask(self):
        return np.hstack([
            [self.env.random_start] * self.env.n_links,  # cos
            [self.env.random_start] * self.env.n_links,  # sin
            [self.env.random_start] * self.env.n_links,  # velocity
            [self.env.initial_via_target is None] * 2,  # x-y coordinates of via point distance
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel
