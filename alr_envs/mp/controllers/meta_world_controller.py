import numpy as np

from alr_envs.mp.controllers.base_controller import BaseController


class MetaWorldController(BaseController):
    """
    A Metaworld Controller. Using position and velocity information from a provided environment,
    the tracking_controller calculates a response based on the desired position and velocity.
    Unlike the other Controllers, this is a special tracking_controller for MetaWorld environments.
    They use a position delta for the xyz coordinates and a raw position for the gripper opening.

    :param env: A position environment
    """

    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        gripper_pos = des_pos[-1]

        cur_pos = env.current_pos[:-1]
        xyz_pos = des_pos[:-1]

        assert xyz_pos.shape == cur_pos.shape, \
            f"Mismatch in dimension between desired position {xyz_pos.shape} and current position {cur_pos.shape}"
        trq = np.hstack([(xyz_pos - cur_pos), gripper_pos])
        return trq
