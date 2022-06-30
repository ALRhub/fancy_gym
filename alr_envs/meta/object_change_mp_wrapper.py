from typing import Tuple, Union

import numpy as np

from alr_envs.mp.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):
    """
    This Wrapper is for environments where merely the goal changes in the beginning
    and no secondary objects or end effectors are altered at the start of an episode.
    You can verify this by executing the code below for your environment id and check if the output is non-zero
    at the same indices.
    ```python
    import alr_envs
    env = alr_envs.make(env_id, 1)
    print(env.reset() - env.reset())
    array([ 0.        ,  0.        ,  0.        ,  0.        ,    !=0 ,
        !=0       , !=0        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0         ,  0         ,  0         ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.])
    ```
    """

    @property
    def context_mask(self) -> np.ndarray:
        # This structure is the same for all metaworld environments.
        # Only the observations which change could differ
        return np.hstack([
            # Current observation
            [False] * 3,  # end-effector position
            [False] * 1,  # normalized gripper open distance
            [False] * 3,  # main object position
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
        return np.hstack([self.env.data.mocap_pos.flatten() / self.env.action_scale, r_close])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        raise NotImplementedError("Velocity cannot be retrieved.")

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
