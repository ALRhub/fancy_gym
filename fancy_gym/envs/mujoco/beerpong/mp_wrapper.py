from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            'phase_generator_kwargs': {
                'learn_tau': True
            },
            'controller_kwargs': {
                'p_gains': np.array([1.5, 5, 2.55, 3, 2., 2, 1.25]),
                'd_gains': np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125]),
            },
            'basis_generator_kwargs': {
                'num_basis': 2,
                'num_basis_zero_start': 2,
            },
        },
        'DMP': {},
        'ProDMP': {},
    }

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [False] * 7,  # joint velocities
            [False] * 3,  # cup_goal_diff_final
            [False] * 3,  # cup_goal_diff_top
            [True] * 2,  # xy position of cup
            # [False]  # env steps
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[0:7].copy()

    # TODO: Fix this
    def episode_callback(self, action: np.ndarray, mp) -> Tuple[np.ndarray, Union[np.ndarray, None], bool]:
        if mp.learn_tau:
            self.release_step = action[0] / self.dt  # Tau value
        return action, None, True

    def set_context(self, context):
        xyz = np.zeros(3)
        xyz[:2] = context
        xyz[-1] = 0.840
        self.model.body_pos[self.cup_table_id] = xyz
        return self.get_observation_from_step(self.get_obs())


class MPWrapper_FixedRelease(MPWrapper):
    mp_config = {
        'ProMP': {
            'phase_generator_kwargs': {
                'tau': 0.62,
            },
            'controller_kwargs': {
                'p_gains': np.array([1.5, 5, 2.55, 3, 2., 2, 1.25]),
                'd_gains': np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125]),
            },
            'basis_generator_kwargs': {
                'num_basis': 2,
                'num_basis_zero_start': 2,
            },
        },
        'DMP': {},
        'ProDMP': {},
    }
