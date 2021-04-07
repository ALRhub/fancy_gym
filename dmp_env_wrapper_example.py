from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.classic_control.utils import make_holereacher_env, make_holereacher_fix_goal_env, make_holereacher_env_pmp
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    n_samples = 1
    n_cpus = 4
    dim = 15

    # env = DmpAsyncVectorEnv([make_viapointreacher_env(i) for i in range(n_cpus)],
    #                         n_samples=n_samples)

    test_env = make_holereacher_env_pmp(0)()

    # params = 1 * np.random.randn(dim)
    params = np.array([[ -0.13106822,  -0.66268577,  -1.37025136,  -1.34813613,
         -0.34040336,  -1.41684643,   2.81882318,  -1.93383471,
         -5.84213385,  -3.8623558 ,  -1.31946267,   3.19346678,
         -9.6581148 ,  -8.27402906,  -0.42374776,  -2.06852054,
          7.21224904,  -6.81061422,  -9.54973119,  -6.18636867,
         -6.82998929,  13.00398992, -18.28106949,  -6.06678165,
          2.79744735]])

    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    rew, info = test_env.rollout(params, render=True)
    print(rew)

    # out = env(params)
    # print(out)
