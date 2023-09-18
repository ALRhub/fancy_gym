from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.]),
                'd_gains': 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.]),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
        'DMP': {},
        'ProDMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.]),
                'd_gains': 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.]),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        if self.random_init:
            return np.hstack([
                [True] * 7,  # joints position
                [False] * 7,  # joints velocity
                [True] * 3,  # position of box
                [True] * 4,  # orientation of box
                [True] * 3,  # position of target
                [True] * 4,  # orientation of target
                # [True] * 1,  # time
            ])

        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [False] * 3,  # position of box
            [False] * 4,  # orientation of box
            [True] * 3,  # position of target
            [True] * 4,  # orientation of target
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()


class ReplanMPWrapper(MPWrapper):
    mp_config = {
        'ProMP': {},
        'DMP': {},
        'ProDMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.]),
                'd_gains': 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.]),
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 0.3,
                'goal_scale': 0.3,
                'auto_scale_basis': True,
                'goal_offset': 1.0,
                'disable_goal': True,
            },
            'basis_generator_kwargs': {
                'num_basis': 5,
                'basis_bandwidth_factor': 3,
            },
            'phase_generator_kwargs': {
                'alpha_phase': 3,
            },
            'black_box_kwargs': {
                'max_planning_times': 4,
                'replanning_schedule': lambda pos, vel, obs, action, t: t % 25 == 0,
                'condition_on_desired': True,
            }
        }
    }
