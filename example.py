from collections import defaultdict

import gym
import numpy as np


def example_mujoco():
    env = gym.make('alr_envs:ALRReacher-v0')
    rewards = 0
    obs = env.reset()

    # number of environment steps
    for i in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            env.render()

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_dmp():
    # env = gym.make("alr_envs:ViaPointReacherDMP-v0")
    env = gym.make("alr_envs:HoleReacherDMP-v0")
    rewards = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            # render full DMP trajectory
            # render can only be called once in the beginning as well. That would render every trajectory
            # Calling it after every trajectory allows to modify the mode. mode=None, disables rendering.
            env.render(mode="human")

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_async(n_cpu=4, seed=int('533D', 16)):
    def make_env(env_id, seed, rank):
        env = gym.make(env_id)
        env.seed(seed + rank)
        return lambda: env

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

    envs = gym.vector.AsyncVectorEnv([make_env("alr_envs:HoleReacherDMP-v0", seed, i) for i in range(n_cpu)])

    obs = envs.reset()
    print(sample(envs, 16))


if __name__ == '__main__':
    # example_mujoco()
    # example_dmp()
    example_async()
