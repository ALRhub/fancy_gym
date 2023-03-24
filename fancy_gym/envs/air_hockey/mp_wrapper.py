import numpy as np
from typing import Union, Tuple
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class PlanarMPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True] * 3,  # puck position [x, y, theta]
            [True] * 3,  # puck velocity [dx, dy, dtheta]
            [True] * 3,  # joint position
            [True] * 3,  # joint velocity
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        # q_pos, _ = self.env.env.get_joints(obs)
        return self.unwrapped.base_env.q_pos_prev
        # return self.unwrapped.robot_data.qpos.copy()
        # return self.unwrapped._data.qpos[:self.dof].copy()
        # return np.array([0, 0, 0])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        # _, q_vel = self.env.env.get_joints(obs)
        return self.unwrapped.base_env.q_vel_prev
        # return self.unwrapped.robot_data.qvel.copy()
        # return self.unwrapped._data.qvel[:self.dof].copy()
        # return np.array([0, 0, 0])
