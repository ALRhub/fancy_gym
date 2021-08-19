from typing import Tuple, Union

import numpy as np

from mp_env_api import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        # This structure is the same for all metaworld environments.
        # Only the observations which change could differ
        return np.hstack([
            # Current observation
            [False] * 3,  # end-effector position
            [False] * 1,  # normalized gripper open distance
            [True] * 3,  # main object position
            [False] * 4,  # main object quaternion
            [False] * 3,  # secondary object position
            [False] * 4,  # secondary object quaternion
            # Previous observation
            # TODO: Include previous values? According to their source they might be wrong for the first iteration.
            [False] * 3,  # previous end-effector position
            [False] * 1,  # previous normalized gripper open distance
            [False] * 3,  # previous main object position
            [False] * 4,  # previous main object quaternion
            [False] * 3,  # previous second object position
            [False] * 4,  # previous second object quaternion
            # Goal
            [True] * 3,  # goal position
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        r_close = self.env.data.get_joint_qpos("r_close")
        return np.hstack([self.env.data.mocap_pos.flatten(), r_close])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        raise NotImplementedError("Velocity cannot be retrieved.")

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
