from typing import Union

import numpy as np

from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class DMCBallInCupMPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        # Besides the ball position, the environment is always set to 0.
        return np.hstack([
            [False] * 2,  # cup position
            [True] * 2,  # ball position
            [False] * 2,  # cup velocity
            [False] * 2,  # ball velocity
        ])

    @property
    def start_pos(self) -> Union[float, int, np.ndarray]:
        return np.hstack([self.physics.named.data.qpos['cup_x'], self.physics.named.data.qpos['cup_z']])

    @property
    def dt(self) -> Union[float, int]:
        # Taken from: https://github.com/deepmind/dm_control/blob/master/dm_control/suite/ball_in_cup.py#L27
        return 0.02
