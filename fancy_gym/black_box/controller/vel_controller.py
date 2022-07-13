from fancy_gym.black_box.controller.base_controller import BaseController


class VelController(BaseController):
    """
    A Velocity Controller. The tracking_controller calculates a response only based on the desired velocity.
    """
    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        return des_vel
