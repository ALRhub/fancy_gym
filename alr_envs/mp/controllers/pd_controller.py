from typing import Union, Tuple

from alr_envs.mp.controllers.base_controller import BaseController


class PDController(BaseController):
    """
    A PD-Controller. Using position and velocity information from a provided environment,
    the tracking_controller calculates a response based on the desired position and velocity

    :param env: A position environment
    :param p_gains: Factors for the proportional gains
    :param d_gains: Factors for the differential gains
    """

    def __init__(self,
                 p_gains: Union[float, Tuple] = 1,
                 d_gains: Union[float, Tuple] = 0.5):
        self.p_gains = p_gains
        self.d_gains = d_gains

    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        assert des_pos.shape == c_pos.shape, \
            f"Mismatch in dimension between desired position {des_pos.shape} and current position {c_pos.shape}"
        assert des_vel.shape == c_vel.shape, \
            f"Mismatch in dimension between desired velocity {des_vel.shape} and current velocity {c_vel.shape}"
        trq = self.p_gains * (des_pos - c_pos) + self.d_gains * (des_vel - c_vel)
        return trq
