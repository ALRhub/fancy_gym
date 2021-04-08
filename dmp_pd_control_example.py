from alr_envs.mujoco.ball_in_a_cup.utils import make_env, make_simple_env
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
    non_vec_env = make_simple_env(0, 0)()

    # params = 0.5 * np.random.randn(dim)
    params = np.array([[11.90777345,  4.47656072, -2.49030537,  2.29386444, -3.5645336 ,
         2.97729181,  4.65224072,  3.72020235,  4.3658366 , -5.8489886 ,
         9.8045112 ,  2.95405854,  4.56178261,  4.70669295,  4.55522522]])

    out2 = non_vec_env.rollout(params, render=True)

    print(out2)
