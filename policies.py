from typing import Tuple, Union

import numpy as np



class BaseController:
    def __init__(self, env, **kwargs):
        self.env = env

    def get_action(self, des_pos, des_vel):
        raise NotImplementedError


class PosController(BaseController):
    """
    A Position Controller. The controller calculates a response only based on the desired position.
    """
    def get_action(self, des_pos, des_vel):
        return des_pos


class VelController(BaseController):
    """
    A Velocity Controller. The controller calculates a response only based on the desired velocity.
    """
    def get_action(self, des_pos, des_vel):
        return des_vel


class PDController(BaseController):
    """
    A PD-Controller. Using position and velocity information from a provided environment,
    the controller calculates a response based on the desired position and velocity

    :param env: A position environment
    :param p_gains: Factors for the proportional gains
    :param d_gains: Factors for the differential gains
    """

    def __init__(self,
                 env,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple]):
        self.p_gains = p_gains
        self.d_gains = d_gains
        super(PDController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        cur_pos = self.env.current_pos
        cur_vel = self.env.current_vel
        assert des_pos.shape == cur_pos.shape, \
            f"Mismatch in dimension between desired position {des_pos.shape} and current position {cur_pos.shape}"
        assert des_vel.shape == cur_vel.shape, \
            f"Mismatch in dimension between desired velocity {des_vel.shape} and current velocity {cur_vel.shape}"
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq


class MetaWorldController(BaseController):
    """
    A Metaworld Controller. Using position and velocity information from a provided environment,
    the controller calculates a response based on the desired position and velocity.
    Unlike the other Controllers, this is a special controller for MetaWorld environments.
    They use a position delta for the xyz coordinates and a raw position for the gripper opening.

    :param env: A position environment
    """

    def __init__(self,
                 env
                 ):
        super(MetaWorldController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        gripper_pos = des_pos[-1]

        cur_pos = self.env.current_pos[:-1]
        xyz_pos = des_pos[:-1]

        assert xyz_pos.shape == cur_pos.shape, \
            f"Mismatch in dimension between desired position {xyz_pos.shape} and current position {cur_pos.shape}"
        trq = np.hstack([(xyz_pos - cur_pos), gripper_pos])
        return trq


#TODO: Do we need this class?
class PDControllerExtend(BaseController):
    """
    A PD-Controller. Using position and velocity information from a provided positional environment,
    the controller calculates a response based on the desired position and velocity

    :param env: A position environment
    :param p_gains: Factors for the proportional gains
    :param d_gains: Factors for the differential gains
    """

    def __init__(self,
                 env,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple]):

        self.p_gains = p_gains
        self.d_gains = d_gains
        super(PDControllerExtend, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        cur_pos = self.env.current_pos
        cur_vel = self.env.current_vel
        if len(des_pos) != len(cur_pos):
            des_pos = self.env.extend_des_pos(des_pos)
        if len(des_vel) != len(cur_vel):
            des_vel = self.env.extend_des_vel(des_vel)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq


def get_policy_class(policy_type, env, **kwargs):
    if policy_type == "motor":
        return PDController(env, **kwargs)
    elif policy_type == "velocity":
        return VelController(env)
    elif policy_type == "position":
        return PosController(env)
    elif policy_type == "metaworld":
        return MetaWorldController(env)
    else:
        raise ValueError(f"Invalid controller type {policy_type} provided. Only 'motor', 'velocity', 'position'  "
                         f"and 'metaworld are currently supported controllers.")
