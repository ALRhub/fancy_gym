from alr_envs.mujoco.ball_in_a_cup.utils import make_simple_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    dim = 15
    n_cpus = 4

    n_samples = 1

    vec_env = DmpAsyncVectorEnv([make_simple_env(i) for i in range(n_cpus)],
                                n_samples=n_samples)

    params = np.tile(1 * np.random.randn(n_samples, dim), (10, 1))

    rewards, infos = vec_env(params)
    print(rewards)
    #
    # non_vec_env = make_simple_env(0, 0)()
    #
    # params = 10 * np.random.randn(dim)

    # out2 = non_vec_env.rollout(params, render=True)

    # print(out2)