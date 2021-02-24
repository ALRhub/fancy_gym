from alr_envs.utils.detpmp_env_wrapper import DetPMPEnvWrapper
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_simple import ALRBallInACupEnv as ALRBallInACupEnvSimple


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
        env = ALRBallInACupEnv()

        env = DetPMPEnvWrapper(env,
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
        env = ALRBallInACupEnvSimple()

        env = DetPMPEnvWrapper(env,
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
        env = ALRBallInACupEnvSimple()

        env = DetPMPEnvWrapper(env,
                               num_dof=3,
                               num_basis=5,
                               width=0.005,
                               policy_type="motor",
                               start_pos=env.start_pos[1::2],
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
