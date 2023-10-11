from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'controller_type': 'velocity',
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 2,
            },
        },
        'DMP': {
            'controller_kwargs': {
                'controller_type': 'velocity',
            },
            'trajectory_generator_kwargs': {
                # TODO: Before it was weight scale 50 and goal scale 0.1. We now only have weight scale and thus set it to 500. Check
                'weights_scale': 500,
            },
            'phase_generator_kwargs': {
                'alpha_phase': 2.5,
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
            [self.env.initial_width is None],  # hole width
            # [self.env.hole_depth is None],  # hole depth
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel
