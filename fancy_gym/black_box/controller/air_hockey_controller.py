import numpy as np
from fancy_gym.black_box.controller.base_controller import BaseController


class AirHockeyController(BaseController):

    def __init__(self, dof=3, **kwargs):
        self.dof = dof

    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        return np.hstack([des_pos, des_vel])
