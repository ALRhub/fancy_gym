import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import numpy as np
from _collections import defaultdict


def make_env(env_id, rank, seed=0, **env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env.seed(seed + rank)
    return lambda: env


def split_array(ary, size):
    n_samples = len(ary)
    if n_samples < size:
        tmp = np.zeros((size, ary.shape[1]))
        tmp[0:n_samples] = ary
        return [tmp]
    elif n_samples == size:
        return [ary]
    else:
        repeat = int(np.ceil(n_samples / size))
        split = [k * size for k in range(1, repeat)]
        sub_arys = np.split(ary, split)

        if n_samples % size != 0:
            tmp = np.zeros_like(sub_arys[0])
            last = sub_arys[-1]
            tmp[0: len(last)] = last
            sub_arys[-1] = tmp

    return sub_arys


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


class AlrMpEnvSampler:
    """
    An asynchronous sampler for non contextual MPWrapper environments. A sampler object can be called with a set of
    parameters and returns the corresponding final obs, rewards, dones and info dicts.
    """
    def __init__(self, env_id, num_envs, seed=0, **env_kwargs):
        self.num_envs = num_envs
        self.env = AsyncVectorEnv([make_env(env_id, seed, i, **env_kwargs) for i in range(num_envs)])

    def __call__(self, params):
        params = np.atleast_2d(params)
        n_samples = params.shape[0]
        split_params = split_array(params, self.num_envs)

        vals = defaultdict(list)
        for p in split_params:
            self.env.reset()
            obs, reward, done, info = self.env.step(p)
            vals['obs'].append(obs)
            vals['reward'].append(reward)
            vals['done'].append(done)
            vals['info'].append(info)

        # do not return values above threshold
        return np.vstack(vals['obs'])[:n_samples], np.hstack(vals['reward'])[:n_samples],\
            _flatten_list(vals['done'])[:n_samples], _flatten_list(vals['info'])[:n_samples]


class AlrContextualMpEnvSampler:
    """
    An asynchronous sampler for non contextual MPWrapper environments. A sampler object can be called with a set of
    parameters and returns the corresponding final obs, rewards, dones and info dicts.
    """
    def __init__(self, env_id, num_envs, seed=0, **env_kwargs):
        self.num_envs = num_envs
        self.env = AsyncVectorEnv([make_env(env_id, seed, i, **env_kwargs) for i in range(num_envs)])

    def __call__(self, dist, n_samples):

        repeat = int(np.ceil(n_samples / self.env.num_envs))
        vals = defaultdict(list)
        for i in range(repeat):
            new_contexts = self.env.reset()
            vals['new_contexts'].append(new_contexts)
            new_samples, new_contexts = dist.sample(new_contexts)
            vals['new_samples'].append(new_samples)

            obs, reward, done, info = self.env.step(new_samples)
            vals['obs'].append(obs)
            vals['reward'].append(reward)
            vals['done'].append(done)
            vals['info'].append(info)

        # do not return values above threshold
        return np.vstack(vals['new_samples'])[:n_samples], np.vstack(vals['new_contexts'])[:n_samples], \
            np.vstack(vals['obs'])[:n_samples], np.hstack(vals['reward'])[:n_samples], \
            _flatten_list(vals['done'])[:n_samples], _flatten_list(vals['info'])[:n_samples]


if __name__ == "__main__":
    env_name = "alr_envs:ALRBallInACupSimpleDMP-v0"
    n_cpu = 8
    dim = 15
    n_samples = 10

    sampler = AlrMpEnvSampler(env_name, num_envs=n_cpu)

    thetas = np.random.randn(n_samples, dim)  # usually form a search distribution

    _, rewards, __, ___ = sampler(thetas)

    print(rewards)
