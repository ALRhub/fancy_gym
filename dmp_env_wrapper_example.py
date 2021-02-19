from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    n_samples = 1
    n_cpus = 4
    dim = 25

    # env = DmpAsyncVectorEnv([make_viapointreacher_env(i) for i in range(n_cpus)],
    #                         n_samples=n_samples)

    test_env = make_viapointreacher_env(0)()

    # params = np.random.randn(n_samples, dim)
    params = np.array([ 217.54494933,   -1.85169983,   24.08414447,   42.23816868,
         23.32071702,    7.60780651,  -31.74777741,  265.50634253,
        463.43822562,  245.93948374, -272.64003621,  -45.24999553,
        503.21185823,  809.17742517,  393.12387021, -196.54196471,
          6.79327307,  374.82429078,  552.4119579 ,  197.3963343 ,
        243.87357056,  -39.56041541, -616.93957463, -710.0772516 ,
       -414.21769789])

    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    rew, info = test_env.rollout(params, render=True)
    print(rew)

    # out = env(params)
    # print(out)
