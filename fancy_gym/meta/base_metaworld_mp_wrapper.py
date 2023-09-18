from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class BaseMetaworldMPWrapper(RawInterfaceWrapper):
    mp_config = {
        'inherit_defaults': False,
        'ProMP': {
            'wrappers': [],
            'trajectory_generator_kwargs': {
                'trajectory_generator_type': 'promp',
                'weights_scale': 10,
            },
            'phase_generator_kwargs': {
                'phase_generator_type': 'linear'
            },
            'controller_kwargs': {
                'controller_type': 'metaworld',
            },
            'basis_generator_kwargs': {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 1
            },
            'black_box_kwargs': {
                'condition_on_desired': False,
            },
        },
        'DMP': {},
        'ProDMP': {
            'wrappers': [],
            'trajectory_generator_kwargs': {
                'trajectory_generator_type': 'prodmp',
                'auto_scale_basis': True,
                'weights_scale': 10,
                # 'goal_scale': 0.,
                'disable_goal': True,
            },
            'phase_generator_kwargs': {
                'phase_generator_type': 'exp',
                # 'alpha_phase' : 3,
            },
            'controller_kwargs': {
                'controller_type': 'metaworld',
            },
            'basis_generator_kwargs': {
                'basis_generator_type': 'prodmp',
                'num_basis': 5,
                'alpha': 10
            },
            'black_box_kwargs': {
                'condition_on_desired': False,
            },
        },
    }

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        r_close = self.env.data.joint('r_close').qpos
        return np.hstack([self.env.data.mocap_pos.flatten() / self.env.action_scale, r_close])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.zeros(4, )
        # raise NotImplementedError('Velocity cannot be retrieved.')
