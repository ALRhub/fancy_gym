from alr_envs.mujoco.ball_in_a_cup.utils import make_simple_dmp_env
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
    params = np.array([-2.63357598, -1.04950296, -0.44330737,  0.52950017,  4.29247739,
        4.52473661, -0.05685977, -0.76796851,  3.71540499,  1.22631059,
        2.20412438,  3.91588129, -0.12652723, -3.0788211 ,  0.56204464])

    out2 = non_vec_env.rollout(params, render=True )

    print(out2)
