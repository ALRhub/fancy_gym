import warnings
from collections import defaultdict

import gym
import numpy as np

from alr_envs.utils.make_env_helpers import make_env
from alr_envs.utils.mp_env_async_sampler import AlrContextualMpEnvSampler, AlrMpEnvSampler, DummyDist


def example_general(env_id: str, seed=1, iterations=1000):
    """
    Example for running any env in the step based setting.
    This also includes DMC environments when leveraging our custom make_env function.
    """

    env = make_env(env_id, seed)
    rewards = 0
    obs = env.reset()
    print("Observation shape: ", env.observation_space.shape)
    print("Action shape: ", env.action_space.shape)

    # number of environment steps
    for i in range(iterations):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            env.render()

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_async(env_id="alr_envs:HoleReacherDMP-v0", n_cpu=4, seed=int('533D', 16)):
    def sample(env: gym.vector.VectorEnv, n_samples=100):
        # for plotting
        rewards = np.zeros(n_cpu)

        # this would generate more samples than requested if n_samples % num_envs != 0
        repeat = int(np.ceil(n_samples / env.num_envs))
        vals = defaultdict(list)
        for i in range(repeat):
            obs, reward, done, info = envs.step(envs.action_space.sample())
            vals['obs'].append(obs)
            vals['reward'].append(reward)
            vals['done'].append(done)
            vals['info'].append(info)
            rewards += reward
            if np.any(done):
                print(rewards[done])
                rewards[done] = 0

        # do not return values above threshold
        return (*map(lambda v: np.stack(v)[:n_samples], vals.values()),)

    from alr_envs.utils.make_env_helpers import make_env_rank
    envs = gym.vector.AsyncVectorEnv([make_env_rank(env_id, seed, i) for i in range(n_cpu)])
    # envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed + i) for i in range(n_cpu)])

    obs = envs.reset()
    print(sample(envs, 16))


if __name__ == '__main__':
    # Mujoco task from framework
    example_general("alr_envs:ALRReacher-v0")
