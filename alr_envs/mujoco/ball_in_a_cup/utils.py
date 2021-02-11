from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapper
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_simple import ALRBallInACupEnv as ALRBallInACupEnvSimple


# TODO: add make_env for standard biac


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

        env = DmpEnvWrapper(env,
                            policy_type="motor",
                            start_pos=env.start_pos[1::2],
                            final_pos=env.start_pos[1::2],
                            num_dof=3,
                            num_basis=8,
                            duration=3.5,
                            alpha_phase=3,
                            post_traj_time=4.5,
                            dt=env.dt,
                            learn_goal=False,
                            weights_scale=50)

        env.seed(seed + rank)
        return env

    return _init
