from alr_envs.utils.mps.detpmp_wrapper import DetPMPWrapper
from alr_envs.mujoco.beerpong.beerpong import ALRBeerpongEnv
from alr_envs.mujoco.beerpong.beerpong_simple import ALRBeerpongEnv as ALRBeerpongEnvSimple


def make_contextual_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        env = ALRBeerpongEnv()

        env = DetPMPWrapper(env, num_dof=7, num_basis=5, width=0.005, duration=3.5, dt=env.dt, post_traj_time=4.5,
                            policy_type="motor", weights_scale=0.5, zero_start=True, zero_goal=True)

        env.seed(seed + rank)
        return env

    return _init


def _make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        env = ALRBeerpongEnvSimple()

        env = DetPMPWrapper(env, num_dof=7, num_basis=5, width=0.005, duration=3.5, dt=env.dt, post_traj_time=4.5,
                            policy_type="motor", weights_scale=0.25, zero_start=True, zero_goal=True)

        env.seed(seed + rank)
        return env

    return _init


def make_simple_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        env = ALRBeerpongEnvSimple()

        env = DetPMPWrapper(env, num_dof=3, num_basis=5, width=0.005, duration=3.5, dt=env.dt, post_traj_time=4.5,
                            policy_type="motor", weights_scale=0.5, zero_start=True, zero_goal=True)

        env.seed(seed + rank)
        return env

    return _init
