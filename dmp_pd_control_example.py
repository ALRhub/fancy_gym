from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapper
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_simple import ALRBallInACupEnv
import numpy as np


if __name__ == "__main__":

    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            _env = ALRBallInACupEnv()

            _env = DmpEnvWrapper(_env,
                                 num_dof=3,
                                 num_basis=8,
                                 duration=3.5,
                                 alpha_phase=3,
                                 dt=_env.dt,
                                 learn_goal=False,
                                 start_pos=_env.start_pos[1::2],
                                 final_pos=_env.start_pos[1::2],
                                 policy_type="motor"
                                 )
            _env.seed(seed + rank)
            return _env
        return _init

    dim = 24

    n_samples = 10

    vec_env = DmpAsyncVectorEnv([make_env(i) for i in range(4)],
                                n_samples=n_samples,
                                context="spawn",
                                shared_memory=False,
                                worker=_worker)

    params = 10 * np.random.randn(n_samples, dim)

    out = vec_env(params)

    non_vec_env = make_env(0, 0)()

    params = 10 * np.random.randn(dim)

    out2 = non_vec_env.rollout(params, render=True)

    print(out2)
