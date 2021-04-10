from alr_envs.mujoco.ball_in_a_cup.utils import make_env, make_simple_env, make_simple_dmp_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    dim = 15
    n_cpus = 4

    # n_samples = 10
    #
    # vec_env = DmpAsyncVectorEnv([make_simple_env(i) for i in range(n_cpus)],
    #                             n_samples=n_samples)
    #
    # params = np.tile(1 * np.random.randn(n_samples, dim), (10, 1))
    #
    # rewards, infos = vec_env(params)
    # print(rewards)
    #
    non_vec_env = make_simple_dmp_env(0, 0)()

    # params = 0.5 * np.random.randn(dim)
    params = np.array([[-2.04114375, -2.62248565,  1.35999138,  4.29883804,  0.09143854,
         8.1752235 , -1.47063842,  0.60865483, -3.1697385 , 10.95458786,
         2.81887935,  3.6400505 ,  1.43011501, -4.36044191, -3.66816722]])

    out2 = non_vec_env.rollout(params, render=False)

    print(out2)
