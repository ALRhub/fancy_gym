from typing import Iterable, Type, Union, Optional

import numpy as np
from gymnasium import register as gym_register

import uuid

import gymnasium as gym
import numpy as np

from fancy_gym.utils.env_compatibility import EnvCompatibility

try:
    import metaworld
except Exception:
    # catch Exception as Import error does not catch missing mujoco-py
    # TODO: Print info?
    pass


class MujocoMapSpacesWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        eos = env.observation_space
        eas = env.action_space

        Obs_Space_Class = getattr(gym.spaces, str(eos.__class__).split("'")[1].split('.')[-1])
        Act_Space_Class = getattr(gym.spaces, str(eas.__class__).split("'")[1].split('.')[-1])

        self.observation_space = Obs_Space_Class(low=eos.low, high=eos.high, dtype=eos.dtype)
        self.action_space = Act_Space_Class(low=eas.low, high=eas.high, dtype=eas.dtype)


class MitigateMetaworldBug(gym.Wrapper, gym.utils.RecordConstructorArgs):
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


def make_metaworld(underlying_id: str, seed: int = 1, render_mode: Optional[str] = None, **kwargs):
    if underlying_id not in metaworld.ML1.ENV_NAMES:
        raise ValueError(f'Specified environment "{underlying_id}" not present in metaworld ML1.')

    _env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[underlying_id + "-goal-observable"](seed=seed, **kwargs)

    # setting this avoids generating the same initialization after each reset
    _env._freeze_rand_vec = False
    # New argument to use global seeding
    _env.seeded_rand_vec = True

    max_episode_steps = _env.max_path_length

    # TODO remove this as soon as there is support for the new API
    _env = EnvCompatibility(_env, render_mode)
    env = _env

    # gym_id = '_metaworld_compat_' + uuid.uuid4().hex + '-v0'
    # gym_register(
    #    id=gym_id,
    #    entry_point=lambda: _env,
    #    max_episode_steps=max_episode_steps,
    # )

    # TODO enable checker when the incorrect dtype of obs and observation space are fixed by metaworld
    # env = gym.make(gym_id, disable_env_checker=True)
    env = MujocoMapSpacesWrapper(env)
    # TODO remove, when this has been fixed upstream
    env = MitigateMetaworldBug(env)
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
