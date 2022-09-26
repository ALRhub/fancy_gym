from itertools import chain
from typing import Tuple, Type, Union, Optional

import gym
import numpy as np
import pytest
from gym import register
from gym.wrappers import TimeLimit

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from gym.core import ActType, ObsType

import fancy_gym

SEED = 1
ENV_IDS = ['Reacher5d-v0', 'dmc:ball_in_cup-catch', 'metaworld:reach-v2', 'Reacher-v2']
WRAPPERS = [fancy_gym.envs.mujoco.reacher.MPWrapper, fancy_gym.dmc.suite.ball_in_cup.MPWrapper,
            fancy_gym.meta.goal_object_change_mp_wrapper.MPWrapper, fancy_gym.open_ai.mujoco.reacher_v2.MPWrapper]
ALL_MP_ENVS = chain(*fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values())


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    dt = 0.01

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
        return np.ones(self.action_space.shape)


@pytest.fixture(scope="session", autouse=True)
def setup():
    register(
        id=f'toy-v0',
        entry_point='test.test_black_box:ToyEnv',
        max_episode_steps=50,
    )


@pytest.mark.parametrize('env_id', ENV_IDS)
def test_missing_wrapper(env_id: str):
    with pytest.raises(ValueError):
        fancy_gym.make_bb(env_id, [], {}, {}, {}, {}, {})


@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
def test_context_space(env_wrap: Tuple[str, Type[RawInterfaceWrapper]]):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    # check if observation space matches with the specified mask values which are true
    env_step = fancy_gym.make(env_id, SEED)
    wrapper = wrapper_class(env_step)
    assert env.observation_space.shape == wrapper.context_mask[wrapper.context_mask].shape


@pytest.mark.parametrize('env_id', ENV_IDS)
@pytest.mark.parametrize('reward_aggregation', [np.sum, np.mean, np.median, lambda x: np.mean(x[::2])])
def test_aggregation(env_id: str, reward_aggregation: callable):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'reward_aggregation': reward_aggregation},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    env.reset()

    # ToyEnv only returns 1 as reward
    assert env.step(env.action_space.sample())[1] == reward_aggregation(np.ones(50, ))


@pytest.mark.parametrize('env_id', ENV_IDS)
@pytest.mark.parametrize('add_time_aware_wrapper_before', [True, False])
def test_learn_sub_trajectories(env_id: str, add_time_aware_wrapper_before: bool):
    env_step = fancy_gym.make(env_id, SEED)
    env = fancy_gym.make_bb(env_id, [], {}, {}, {}, {'phase_generator_type': 'linear'}, {})

    # has time aware wrapper
    if add_time_aware_wrapper_before:
        pass

    assert env.learn_sub_trajectories
    assert env.learn_tau
    assert env.observation_space == env_step.observation_space
