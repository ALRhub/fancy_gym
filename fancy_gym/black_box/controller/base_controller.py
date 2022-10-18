class BaseController:

    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        raise NotImplementedError

    def __call__(self, des_pos, des_vel, c_pos, c_vel):
        return self.get_action(des_pos, des_vel, c_pos, c_vel)
