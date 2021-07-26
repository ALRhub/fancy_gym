from typing import Tuple, Union

import numpy as np

from mp_env_api.interface_wrappers.mp_env_wrapper import MPEnvWrapper


class DMCReachSiteMPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
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
        return self.env.dt
