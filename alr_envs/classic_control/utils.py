from alr_envs.classic_control.hole_reacher import HoleReacher
from alr_envs.classic_control.viapoint_reacher import ViaPointReacher
from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapper


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
        _env = ViaPointReacher(num_links=5,
                               allow_self_collision=False,
                               collision_penalty=1000)

        _env = DmpEnvWrapper(_env,
                             num_dof=5,
                             num_basis=5,
                             duration=2,
                             alpha_phase=2,
                             dt=_env.dt,
                             start_pos=_env.start_pos,
                             learn_goal=False,
                             policy_type="velocity")
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
        _env = HoleReacher(num_links=5,
                           allow_self_collision=False,
                           allow_wall_collision=False,
                           hole_width=0.15,
                           hole_depth=1,
                           hole_x=1,
                           collision_penalty=100000)

        _env = DmpEnvWrapper(_env,
                             num_dof=5,
                             num_basis=5,
                             duration=2,
                             dt=_env.dt,
                             learn_goal=True,
                             alpha_phase=2,
                             start_pos=_env.start_pos,
                             policy_type="velocity",
                             weights_scale=100,
                             )
        _env.seed(seed + rank)
        return _env

    return _init
