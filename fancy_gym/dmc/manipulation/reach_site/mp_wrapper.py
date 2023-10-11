from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 50.0,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 0.2,
            },
        },
        'DMP': {
            'controller_kwargs': {
                'p_gains': 50.0,
            },
            'phase_generator': {
                'alpha_phase': 2,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 500,
            },
        },
        'ProDMP': {},
    }

    @property
    def context_mask(self) -> np.ndarray:
        # Joint and target positions are randomized, velocities are always set to 0.
        return np.hstack([
            [True] * 3,  # target position
            [True] * 12,  # sin/cos arm joint position
            [True] * 6,  # arm joint torques
            [False] * 6,  # arm joint velocities
            [True] * 3,  # sin/cos hand joint position
            [False] * 3,  # hand joint velocities
            [True] * 3,  # hand pinch site position
            [True] * 9,  # pinch site rmat
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.physics.named.data.qpos[:]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.physics.named.data.qvel[:]

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.control_timestep()
