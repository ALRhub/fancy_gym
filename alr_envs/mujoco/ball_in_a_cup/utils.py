from alr_envs.utils.wrapper.detpmp_wrapper import DetPMPWrapper
from alr_envs.utils.wrapper.dmp_wrapper import DmpWrapper
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv


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
        env = ALRBallInACupEnv(reward_type="contextual_goal")

        env = DetPMPWrapper(env,
                            num_dof=7,
                            num_basis=5,
                            width=0.005,
                            policy_type="motor",
                            start_pos=env.start_pos,
                            duration=3.5,
                            post_traj_time=4.5,
                            dt=env.dt,
                            weights_scale=0.5,
                            zero_start=True,
                            zero_goal=True
                            )

        env.seed(seed + rank)
        return env

    return _init


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
        env = ALRBallInACupEnv(reward_type="simple")

        env = DetPMPWrapper(env,
                            num_dof=7,
                            num_basis=5,
                            width=0.005,
                            policy_type="motor",
                            start_pos=env.start_pos,
                            duration=3.5,
                            post_traj_time=4.5,
                            dt=env.dt,
                            weights_scale=0.2,
                            zero_start=True,
                            zero_goal=True
                            )

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
        env = ALRBallInACupEnv(reward_type="simple")

        env = DetPMPWrapper(env,
                            num_dof=3,
                            num_basis=5,
                            width=0.005,
                            off=-0.1,
                            policy_type="motor",
                            start_pos=env.start_pos[1::2],
                            duration=3.5,
                            post_traj_time=4.5,
                            dt=env.dt,
                            weights_scale=0.25,
                            zero_start=True,
                            zero_goal=True
                            )

        env.seed(seed + rank)
        return env

    return _init


def make_simple_dmp_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :returns a function that generates an environment
    """

    def _init():
        _env = ALRBallInACupEnv(reward_type="simple")

        _env = DmpWrapper(_env,
                          num_dof=3,
                          num_basis=5,
                          duration=3.5,
                          post_traj_time=4.5,
                          bandwidth_factor=2.5,
                          dt=_env.dt,
                          learn_goal=False,
                          alpha_phase=3,
                          start_pos=_env.start_pos[1::2],
                          final_pos=_env.start_pos[1::2],
                          policy_type="motor",
                          weights_scale=100,
                          )

        _env.seed(seed + rank)
        return _env

    return _init
