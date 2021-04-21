import alr_envs.classic_control.hole_reacher as hr
import alr_envs.classic_control.viapoint_reacher as vpr
from alr_envs.utils.wrapper.dmp_wrapper import DmpWrapper
from alr_envs.utils.wrapper.detpmp_wrapper import DetPMPWrapper
import numpy as np


def make_viapointreacher_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        _env = vpr.ViaPointReacher(n_links=5,
                               allow_self_collision=False,
                               collision_penalty=1000)

        _env = DmpWrapper(_env,
                          num_dof=5,
                          num_basis=5,
                          duration=2,
                          alpha_phase=2.5,
                          dt=_env.dt,
                          start_pos=_env.start_pos,
                          learn_goal=False,
                          policy_type="velocity",
                          weights_scale=50)
        _env.seed(seed + rank)
        return _env

    return _init


def make_holereacher_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        _env = hr.HoleReacher(n_links=5,
                           allow_self_collision=False,
                           allow_wall_collision=False,
                           hole_width=0.25,
                           hole_depth=1,
                           hole_x=2,
                           collision_penalty=100)

        _env = DmpWrapper(_env,
                          num_dof=5,
                          num_basis=5,
                          duration=2,
                          bandwidth_factor=2,
                          dt=_env.dt,
                          learn_goal=True,
                          alpha_phase=2,
                          start_pos=_env.start_pos,
                          policy_type="velocity",
                          weights_scale=50,
                          goal_scale=0.1
                          )

        _env.seed(seed + rank)
        return _env

    return _init


def make_holereacher_fix_goal_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        _env = hr.HoleReacher(n_links=5,
                           allow_self_collision=False,
                           allow_wall_collision=False,
                           hole_width=0.15,
                           hole_depth=1,
                           hole_x=1,
                           collision_penalty=100)

        _env = DmpWrapper(_env,
                          num_dof=5,
                          num_basis=5,
                          duration=2,
                          dt=_env.dt,
                          learn_goal=False,
                          final_pos=np.array([2.02669572, -1.25966385, -1.51618198, -0.80946476,  0.02012344]),
                          alpha_phase=2,
                          start_pos=_env.start_pos,
                          policy_type="velocity",
                          weights_scale=50,
                          goal_scale=1
                          )

        _env.seed(seed + rank)
        return _env

    return _init


def make_holereacher_env_pmp(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        _env = hr.HoleReacher(n_links=5,
                           allow_self_collision=False,
                           allow_wall_collision=False,
                           hole_width=0.15,
                           hole_depth=1,
                           hole_x=1,
                           collision_penalty=1000)

        _env = DetPMPWrapper(_env,
                             num_dof=5,
                             num_basis=5,
                             width=0.02,
                             policy_type="velocity",
                             start_pos=_env.start_pos,
                             duration=2,
                             post_traj_time=0,
                             dt=_env.dt,
                             weights_scale=0.2,
                             zero_start=True,
                             zero_goal=False
                             )
        _env.seed(seed + rank)
        return _env

    return _init
