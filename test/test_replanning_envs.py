from itertools import chain
from typing import Tuple, Type, Union, Optional, Callable

import gym
import numpy as np
import pytest
from gym import register
from gym.core import ActType, ObsType

import fancy_gym
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper

import fancy_gym
from test.utils import run_env, run_env_determinism

Fancy_ProDMP_IDS = fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS['ProDMP']

All_ProDMP_IDS = fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS['ProDMP']


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
        return np.array([-1])

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return np.array([-1]), 1, False, {}

    def render(self, mode="human"):
        pass


class ToyWrapper(RawInterfaceWrapper):

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.ones(self.action_space.shape)

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.zeros(self.action_space.shape)

@pytest.fixture(scope="session", autouse=True)
def setup():
    register(
        id=f'toy-v0',
        entry_point='test.test_black_box:ToyEnv',
        max_episode_steps=50,
    )
# @pytest.mark.parametrize('env_id', All_ProDMP_IDS)
# def test_replanning_envs(env_id: str):
#     """Tests that ProDMP environments run without errors using random actions."""
#     run_env(env_id)
#
# @pytest.mark.parametrize('env_id', All_ProDMP_IDS)
# def test_replanning_determinism(env_id: str):
#     """Tests that ProDMP environments are deterministic."""
#     run_env_determinism(env_id, 0)

@pytest.mark.parametrize('mp_type', ['promp', 'dmp', 'prodmp'])
def test_missing_local_state(mp_type: str):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'

    env = fancy_gym.make_bb('toy-v0', [RawInterfaceWrapper], {},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': basis_generator_type})
    env.reset()
    with pytest.raises(NotImplementedError):
        env.step(env.action_space.sample())