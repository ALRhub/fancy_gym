from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.classic_control.utils import make_holereacher_env, make_holereacher_fix_goal_env, make_holereacher_env_pmp
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv
import numpy as np


if __name__ == "__main__":

    n_samples = 1
    n_cpus = 4
    dim = 30

    # env = DmpAsyncVectorEnv([make_viapointreacher_env(i) for i in range(n_cpus)],
    #                         n_samples=n_samples)

    test_env = make_holereacher_env(0)()

    # params = 1 * np.random.randn(dim)
    params = np.array([ -1.09434772,   7.09294269,   0.98756352,   1.61950682,
         2.66567135,   1.71267901,   8.20010847,   2.50496653,
        -0.34886972,   2.07807773,   8.68615904,   3.66578556,
         5.24572097,  -3.21506848,  -0.28593896,  17.03756855,
        -5.88445032,   6.02197609,  -3.73457261,  -4.24772663,
         8.69382861, -10.99939646,   5.31356886,   8.57420996,
         1.05616879,  19.79831628, -23.53288774,  -3.32974082,
        -5.86463784,  -9.68133089])


    # params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    rew, info = test_env.rollout(params, render=True)
    print(rew)

    # out = env(params)
    # print(out)
