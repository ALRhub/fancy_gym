from alr_envs.mujoco.ball_in_a_cup.utils import make_simple_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    dim = 24
    n_cpus = 4

    n_samples = 10

    vec_env = DmpAsyncVectorEnv([make_simple_env(i) for i in range(n_cpus)],
                                n_samples=n_samples)

    # params = 10 * np.random.randn(n_samples, dim)
    params = np.array([[ -4.51280364,  24.43701373,  15.73282129, -12.13020392,
         -8.57305795,   2.79806606,  -6.38613201,   5.99309385,
         -2.05631886,  24.71684748,  14.05989949, -14.60456967,
         10.51933419,  -2.43715355,  -6.0767578 ,  13.06498129,
          6.18038374,  11.4153859 ,   1.40753639,   5.57082387,
          9.81989309,   3.60558787,  -9.66996754,  14.28519904]])

    out = vec_env(params)
    print(out)
    #
    non_vec_env = make_simple_env(0, 0)()
    #
    # params = 10 * np.random.randn(dim)

    out2 = non_vec_env.rollout(params, render=True)

    print(out2)
