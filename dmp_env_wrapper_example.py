from alr_envs.utils.dmp_env_wrapper import DmpEnvWrapper
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv, _worker
from alr_envs.classic_control.hole_reacher import HoleReacher
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
            _env = HoleReacher(num_links=5,
                               allow_self_collision=False,
                               allow_wall_collision=False,
                               hole_width=0.15,
                               hole_depth=1,
                               hole_x=1)

            _env = DmpEnvWrapper(_env,
                                 num_dof=5,
                                 num_basis=5,
                                 duration=2,
                                 dt=_env.dt,
                                 learn_goal=True,
                                 alpha_phase=2,
                                 start_pos=_env.start_pos,
                                 policy_type="velocity"
                                 )
            _env.seed(seed + rank)
            return _env
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
    out = env(params)

    print(out)
