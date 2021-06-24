from mp_env_api.mp_wrappers.dmp_wrapper import DmpWrapper
from mp_env_api.mp_wrappers.detpmp_wrapper import DetPMPWrapper
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
    for wrapper in kwargs.pop("wrappers"):
        _env = wrapper(_env)
    return DmpWrapper(_env, **kwargs.get("mp_kwargs"))


def make_detpmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    for wrapper in kwargs.pop("wrappers"):
        _env = wrapper(_env)
    return DetPMPWrapper(_env, **kwargs.get("mp_kwargs"))
