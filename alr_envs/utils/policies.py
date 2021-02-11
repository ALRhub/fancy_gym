from alr_envs.mujoco.alr_mujoco_env import AlrMujocoEnv


class BaseController:
    def __init__(self, env: AlrMujocoEnv):
        self.env = env

    def get_action(self, des_pos, des_vel):
        raise NotImplementedError


class PosController(BaseController):
    def get_action(self, des_pos, des_vel):
        return des_pos


class VelController(BaseController):
    def get_action(self, des_pos, des_vel):
        return des_vel


class PDController(BaseController):
    def __init__(self, env):
        self.p_gains = env.p_gains
        self.d_gains = env.d_gains
        super(PDController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        # TODO: make standardized ALRenv such that all of them have current_pos/vel attributes
        cur_pos = self.env.current_pos
        cur_vel = self.env.current_vel
        if len(des_pos) != len(cur_pos):
            des_pos = self.env.extend_des_pos(des_pos)
        if len(des_vel) != len(cur_vel):
            des_vel = self.env.extend_des_vel(des_vel)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq
