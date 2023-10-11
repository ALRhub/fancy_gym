from typing import Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            "trajectory_generator_kwargs": {
                'trajectory_generator_type': 'promp'
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear'
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": 0.6,
                "d_gains": 0.075,
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 6,
                'num_basis_zero_start': 1
            }
        },
        'DMP': {},
        'ProDMP': {},
    }

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
