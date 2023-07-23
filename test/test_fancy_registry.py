from typing import Tuple, Type, Union, Optional, Callable

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import make
from gymnasium.core import ActType, ObsType

import fancy_gym
from fancy_gym import register

ENV_IDS = ['Reacher5d-v0', 'dm_control/ball_in_cup-catch-v0', 'metaworld/reach-v2', 'Reacher-v2']


class Object(object):
    pass


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    dt = 0.02

    def __init__(self, a: int = 0, b: float = 0.0, c: list = [], d: dict = {}, e: Object = Object()):
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None) -> Union[ObsType, Tuple[ObsType, dict]]:
        obs, options = np.array([-1]), {}
        return obs, options

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, terminated, truncated, info = np.array([-1]), 1, False, False, {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass


@pytest.fixture(scope="session", autouse=True)
def setup():
    register(
        id=f'toy2-v0',
        entry_point='test.test_black_box:ToyEnv',
        max_episode_steps=50,
    )


@pytest.mark.parametrize('env_id', ENV_IDS)
@pytest.mark.parametrize('mp_type', ['promp', 'dmp', 'prodmp'])
def test_make_mp(env_id: str, mp_type: str):
    parts = id.split('-')
    assert len(parts) >= 2 and parts[-1].startswith('v'), 'Malformed env id, must end in -v{int}.'
    fancy_id = '-'.join(parts[:-1]+[mp_type, parts[-1]])

    make(fancy_id)


def test_make_raw_toy():
    make('toy2-v0')


@pytest.mark.parametrize('mp_type', ['promp', 'dmp', 'prodmp'])
def test_make_mp_toy(mp_type: str):
    parts = id.split('-')
    assert len(parts) >= 2 and parts[-1].startswith('v'), 'Malformed env id, must end in -v{int}.'
    fancy_id = '-'.join(['toy2', mp_type, 'v0'])

    make(fancy_id)
