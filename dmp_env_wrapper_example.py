from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.classic_control.utils import make_holereacher_env, make_holereacher_fix_goal_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    n_samples = 1
    n_cpus = 4
    dim = 30

    # env = DmpAsyncVectorEnv([make_viapointreacher_env(i) for i in range(n_cpus)],
    #                         n_samples=n_samples)

    test_env = make_holereacher_env(0)()

    # params = np.random.randn(n_samples, dim)
    params = np.array([  0.57622273,   0.98294602,   1.48964131,   0.65430972,
        -0.26028221,   4.84693322,   1.77366128,   0.51080511,
        -2.38201107,  -0.84990048,   1.02289828,   1.20675551,
         0.38075566,  -1.84282938,  -3.48690172,   2.17434711,
        -1.79285349,  -1.7533641 ,   0.62802966,   1.18928357,
         0.2818753 ,  -3.27708291,  -0.91761804,  -0.38350967,
         2.25849139,  21.57786524, -14.38494647, -11.5380005 ,
       -11.09529721,  -0.39453533])

    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    rew, info = test_env.rollout(params, render=True)
    print(rew)

    # out = env(params)
    # print(out)
