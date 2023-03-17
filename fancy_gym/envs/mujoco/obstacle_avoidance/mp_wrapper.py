from typing import Union, Tuple

import numpy as np
import mujoco

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 1,  # goal x position
            [False] * 1,  # tip to goal distance
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        # return np.zeros(2)
        return self.data.body("rod_tip").xpos[:2].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.zeros(2)

    # def set_context(self, context):
    #     # rest box to initial position
    #     self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)
    #     box_init_pos = np.array([0.4, 0.3, -0.01, 0.0, 0.0, 0.0, 1.0])
    #     self.data.joint("box_joint").qpos = box_init_pos
    # 
    #     self.model.body_pos[2] = context[:3]
    #     self.model.body_quat[2] = context[-4:]
    #     self.model.body_pos[3] = context[:3]
    #     self.model.body_quat[3] = context[-4:]
    # 
    #     # set the robot to the right configuration (rod tip in the box)
    #     desired_tcp_pos = box_init_pos[:3] + np.array([0.0, 0.0, 0.15])
    #     desired_tcp_quat = np.array([0, 1, 0, 0])
    #     desired_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)
    #     self.data.qpos[:7] = desired_joint_pos
    # 
    #     mujoco.mj_forward(self.model, self.data)
    #     self._steps = 0
    #     self._episode_energy = 0.
    # 
    #     return self.create_observation()
