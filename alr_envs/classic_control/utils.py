from alr_envs.classic_control.hole_reacher import HoleReacher
from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapperVel


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        env = HoleReacher(num_links=5,
                          allow_self_collision=False,
                          allow_wall_collision=False,
                          hole_width=0.15,
                          hole_depth=1,
                          hole_x=1,
                          collision_penalty=100000)

        env = DmpEnvWrapperVel(env,
                               num_dof=5,
                               num_basis=5,
                               duration=2,
                               dt=env.dt,
                               learn_goal=True)
        env.seed(seed + rank)
        return env

    return _init
