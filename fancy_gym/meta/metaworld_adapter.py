import random
from typing import Iterable, Type, Union, Optional

import numpy as np
from gymnasium import register as gym_register

import uuid

import gymnasium as gym
import numpy as np

from fancy_gym.utils.env_compatibility import EnvCompatibility

import metaworld

class FixMetaworldHasIncorrectObsSpaceWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        eos = env.observation_space
        eas = env.action_space

        Obs_Space_Class = getattr(gym.spaces, str(eos.__class__).split("'")[1].split('.')[-1])
        Act_Space_Class = getattr(gym.spaces, str(eas.__class__).split("'")[1].split('.')[-1])

        self.observation_space = Obs_Space_Class(low=eos.low-np.inf, high=eos.high+np.inf, dtype=eos.dtype)
        self.action_space = Act_Space_Class(low=eas.low, high=eas.high, dtype=eas.dtype)


class FixMetaworldIncorrectResetPathLengthWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        head = self.env
        try:
            for i in range(16):
                head.curr_path_length = 0
                head = head.env
        except:
            pass
        return ret


class FixMetaworldIgnoresSeedOnResetWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            print('[Fancy Gym] You just called .reset on a Metaworld env and supplied a seed. Metaworld curretly does not correctly implement seeding. Do not rely on deterministic behavior.')
            self.env.seed(kwargs['seed'])
        return self.env.reset(**kwargs)


class FixMetaworldRenderOnStep(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.render_active = False

    def render(self, *args, **kwargs):
        self.render_active = True
        return self.env.render(*args, **kwargs)

    def step(self, *args, **kwargs):
        ret = self.env.step(*args, **kwargs)
        if self.render_active:
            self.env.render()
        return ret


def make_metaworld(underlying_id: str, seed: int = 1, render_mode: Optional[str] = None, **kwargs):
    if underlying_id not in metaworld.ML1.ENV_NAMES:
        raise ValueError(f'Specified environment "{underlying_id}" not present in metaworld ML1.')

    env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[underlying_id + "-goal-observable"](seed=seed, render_mode=render_mode, **kwargs)

    # setting this avoids generating the same initialization after each reset
    env._freeze_rand_vec = False
    # New argument to use global seeding
    env.seeded_rand_vec = True

    env = FixMetaworldHasIncorrectObsSpaceWrapper(env)
    # env = FixMetaworldIncorrectResetPathLengthWrapper(env)
    env = FixMetaworldRenderOnStep(env)
    env = FixMetaworldIgnoresSeedOnResetWrapper(env)
    return env


def register_all_ML1(**kwargs):
    for env_id in metaworld.ML1.ENV_NAMES:
        _env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id + "-goal-observable"](seed=0)
        max_episode_steps = _env.max_path_length

        gym_register(
            id='metaworld/'+env_id,
            entry_point=make_metaworld,
            max_episode_steps=max_episode_steps,
            kwargs={
                'underlying_id': env_id
            },
            **kwargs
        )
