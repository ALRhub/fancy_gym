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
    params = np.array([[1.386102, -3.29980525, 4.70402733, 1.3966668, 0.73774902,
                        3.14676681, -4.98644416, 6.20303193, 1.30502127, -0.09330522,
                        7.62656797, -5.76893033, 3.4706711, -0.6944142, -3.33442788,
                        12.31421548, -0.72760271, -6.9090723, 7.02903814, -8.7236836,
                        1.4805914, 0.53185824, -5.46626893, 0.69692163, 13.58472666,
                        0.77199316, 2.02906724, -3.0203244, -1.00533159, -0.57417351]])

    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    rew, info = test_env.rollout(params, render=True)
    print(rew)

    # out = env(params)
    # print(out)
