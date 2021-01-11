from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapperVel
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.classic_control.hole_reacher import HoleReacher
from gym.vector.async_vector_env import AsyncVectorEnv
import numpy as np


# env = gym.make('alr_envs:SimpleReacher-v0')
# env = HoleReacher(num_links=5,
#                   allow_self_collision=False,
#                   allow_wall_collision=True,
#                   hole_width=0.15,
#                   hole_depth=1,
#                   hole_x=1)
#
# env = DmpEnvWrapperVel(env,
#                        num_dof=5,
#                        num_basis=5,
#                        duration=2,
#                        dt=env._dt,
#                        learn_goal=True)
#
# params = np.hstack([50 * np.random.randn(25), np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4])])
#
# print(params)
#
# env.reset()
# obs, rew, done, info = env.step(params, render=True)
#
# print(env.env._joint_angles)
#
# print(rew)

if __name__ == "__main__":

    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = HoleReacher(num_links=5,
                              allow_self_collision=False,
                              allow_wall_collision=False,
                              hole_width=0.15,
                              hole_depth=1,
                              hole_x=1)

            env = DmpEnvWrapperVel(env,
                                   num_dof=5,
                                   num_basis=5,
                                   duration=2,
                                   dt=env._dt,
                                   learn_goal=True)
            env.seed(seed + rank)
            return env
        return _init

    n_samples = 4

    env = DmpAsyncVectorEnv([make_env(i) for i in range(4)],
                            n_samples=n_samples,
                            context="spawn",
                            shared_memory=False,
                            worker=_worker)

    # params = np.random.randn(4, 25)
    params = np.hstack([50 * np.random.randn(n_samples, 25), np.tile(np.array([np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4]), [n_samples, 1])])

    # env.reset()
    out = env.rollout(params)

    print(out)
