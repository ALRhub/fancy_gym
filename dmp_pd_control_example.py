from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapperPD
from alr_envs.utils.policies import PDController
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.mujoco.reacher.alr_reacher import ALRReacherEnv
import numpy as np


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
            p_gains = np.array([100, 100, 100, 100, 100])
            d_gains = np.array([1, 1, 1, 1, 1])
            policy = PDController(p_gains, d_gains)

            env = ALRReacherEnv()

            env = DmpEnvWrapperPD(env,
                                  num_dof=5,
                                  num_basis=5,
                                  duration=4,
                                  dt=env.dt,
                                  learn_goal=False,
                                  start_pos=env.init_qpos[:5],
                                  final_pos=env.init_qpos[:5],
                                  alpha_phase=2,
                                  policy=policy
                                  )
            env.seed(seed + rank)
            return env
        return _init

    dim = 25
    env = make_env(0, 0)()

    params = 10 * np.random.randn(dim)

    out = env.rollout(params, render=True)

    print(out)
