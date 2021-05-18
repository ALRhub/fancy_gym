from alr_envs.utils.mps.dmp_wrapper import DmpWrapper
from alr_envs.utils.mps.detpmp_wrapper import DetPMPWrapper
import gym
from gym.vector.utils import write_to_shared_memory
import sys


def make_env(env_id, seed, rank):
    env = gym.make(env_id)
    env.seed(seed + rank)
    return lambda: env


def make_contextual_env(env_id, context, seed, rank):
    env = gym.make(env_id, context=context)
    env.seed(seed + rank)
    return lambda: env


def make_dmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    return DmpWrapper(_env, **kwargs)


def make_detpmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    return DetPMPWrapper(_env, **kwargs)
