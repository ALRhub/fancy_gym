from typing import Tuple, Type, Union, Optional, Callable

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import make
from gymnasium.core import ActType, ObsType

import fancy_gym
from fancy_gym import register

KNOWN_NS = ['dm_control', 'fancy', 'metaworld', 'gym']


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

    def render(self):
        pass


@pytest.fixture(scope="session", autouse=True)
def setup():
    register(
        id=f'dummy/toy2-v0',
        entry_point='test.test_black_box:ToyEnv',
        max_episode_steps=50,
    )


@pytest.mark.parametrize('env_id', ['dummy/toy2-v0'])
@pytest.mark.parametrize('mp_type', ['ProMP', 'DMP', 'ProDMP'])
def test_make_mp(env_id: str, mp_type: str):
    parts = env_id.split('/')
    if len(parts) == 1:
        ns, name = 'gym', parts[0]
    elif len(parts) == 2:
        ns, name = parts[0], parts[1]
    else:
        raise ValueError('env id can not contain multiple "/".')

    fancy_id = f'{ns}_{mp_type}/{name}'

    make(fancy_id)


def test_make_raw_toy():
    make('dummy/toy2-v0')


@pytest.mark.parametrize('mp_type', ['ProMP', 'DMP', 'ProDMP'])
def test_make_mp_toy(mp_type: str):
    fancy_id = f'dummy_{mp_type}/toy2-v0'

    make(fancy_id)


@pytest.mark.parametrize('ns', KNOWN_NS)
def test_ns_nonempty(ns):
    assert len(fancy_gym.MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS[ns]), f'The namespace {ns} is empty even though, it should not be...'
