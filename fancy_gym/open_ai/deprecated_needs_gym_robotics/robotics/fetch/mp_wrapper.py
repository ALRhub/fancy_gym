from typing import Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def active_obs(self):
        return np.hstack([
            [False] * 3,  # achieved goal
            [True] * 3,  # desired/true goal
            [False] * 3,  # grip pos
            [True, True, False] * int(self.has_object),  # object position
            [True, True, False] * int(self.has_object),  # object relative position
            [False] * 2,  # gripper state
            [False] * 3 * int(self.has_object),  # object rotation
            [False] * 3 * int(self.has_object),  # object velocity position
            [False] * 3 * int(self.has_object),  # object velocity rotation
            [False] * 3,  # grip velocity position
            [False] * 2,  # gripper velocity
        ]).astype(bool)

    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        # gripper state should be symmetric for left and right.
        # They are controlled with only one action for both gripper joints
        gripper_state = self.sim.data.get_joint_qvel('robot0:r_gripper_finger_joint') * dt
        return np.hstack([grip_velp, gripper_state])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        # gripper state should be symmetric for left and right.
        # They are controlled with only one action for both gripper joints
        gripper_state = self.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        return np.hstack([grip_pos, gripper_state])

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
