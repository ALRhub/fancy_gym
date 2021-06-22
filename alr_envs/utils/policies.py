from typing import Tuple, Union

from gym import Env

from alr_envs.utils.positional_env import PositionalEnv


class BaseController:
    def __init__(self, env: Env, **kwargs):
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
    """
    A PD-Controller. Using position and velocity information from a provided positional environment,
    the controller calculates a response based on the desired position and velocity

    :param env: A position environment
    :param p_gains: Factors for the proportional gains
    :param d_gains: Factors for the differential gains
    """
    def __init__(self,
                 env: PositionalEnv,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple]):
        self.p_gains = p_gains
        self.d_gains = d_gains
        super(PDController, self).__init__(env, )

    def get_action(self, des_pos, des_vel):
        cur_pos = self.env.current_pos
        cur_vel = self.env.current_vel
        assert des_pos.shape != cur_pos.shape, \
            "Mismatch in dimension between desired position {} and current position {}".format(des_pos.shape, cur_pos.shape)
        assert des_vel.shape != cur_vel.shape, \
            "Mismatch in dimension between desired velocity {} and current velocity {}".format(des_vel.shape,
                                                                                               cur_vel.shape)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq


def get_policy_class(policy_type, env, mp_kwargs, **kwargs):
    if policy_type == "motor":
        return PDController(env, p_gains=mp_kwargs['p_gains'], d_gains=mp_kwargs['d_gains'])
    elif policy_type == "velocity":
        return VelController(env)
    elif policy_type == "position":
        return PosController(env)
