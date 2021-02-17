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

    params = np.random.randn(n_samples, dim)
    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    test_env.rollout(params, render=True)

    # out = env(params)
    # print(out)
