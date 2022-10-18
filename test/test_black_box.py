from itertools import chain
from typing import Tuple, Type, Union, Optional

import gym
import numpy as np
import pytest
from gym import register
from gym.core import ActType, ObsType

import fancy_gym
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.time_aware_observation import TimeAwareObservation

SEED = 1
ENV_IDS = ['Reacher5d-v0', 'dmc:ball_in_cup-catch', 'metaworld:reach-v2', 'Reacher-v2']
WRAPPERS = [fancy_gym.envs.mujoco.reacher.MPWrapper, fancy_gym.dmc.suite.ball_in_cup.MPWrapper,
            fancy_gym.meta.goal_object_change_mp_wrapper.MPWrapper, fancy_gym.open_ai.mujoco.reacher_v2.MPWrapper]
ALL_MP_ENVS = chain(*fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values())


class Object(object):
    pass


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    dt = 0.01

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


def test_missing_local_state():
    env = fancy_gym.make_bb('toy-v0', [RawInterfaceWrapper], {},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    with pytest.raises(NotImplementedError):
        env.step(env.action_space.sample())


@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
@pytest.mark.parametrize('verbose', [1, 2])
def test_verbosity(env_wrap: Tuple[str, Type[RawInterfaceWrapper]], verbose: int):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    info_keys = env.step(env.action_space.sample())[3].keys()

    env_step = fancy_gym.make(env_id, SEED)
    env_step.reset()
    info_keys_step = env_step.step(env_step.action_space.sample())[3].keys()

    assert info_keys_step in info_keys
    assert 'trajectory_length' in info_keys

    if verbose >= 2:
        mp_keys = ['position', 'velocities', 'step_actions', 'step_observations', 'step_rewards']
        assert mp_keys in info_keys


@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
def test_length(env_wrap: Tuple[str, Type[RawInterfaceWrapper]]):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    length = env.step(env.action_space.sample())[3]['trajectory_length']

    assert length == env.spec.max_episode_steps


@pytest.mark.parametrize('reward_aggregation', [np.sum, np.mean, np.median, lambda x: np.mean(x[::2])])
def test_aggregation(reward_aggregation: callable):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'reward_aggregation': reward_aggregation},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    # ToyEnv only returns 1 as reward
    assert env.step(env.action_space.sample())[1] == reward_aggregation(np.ones(50, ))


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


@pytest.mark.parametrize('num_dof', [0, 1, 2, 5])
@pytest.mark.parametrize('num_basis', [0, 1, 2, 5])
@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
def test_action_space(num_dof: int, num_basis: int, learn_tau: bool, learn_delay: bool):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {},
                            {'trajectory_generator_type': 'promp',
                             'action_dim': num_dof
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear',
                             'learn_tau': learn_tau,
                             'learn_delay': learn_delay
                             },
                            {'basis_generator_type': 'rbf',
                             'num_basis': num_basis
                             })
    assert env.action_space.shape[0] == num_dof * num_basis + int(learn_tau) + int(learn_delay)


@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [1.0])
@pytest.mark.parametrize('c', [[1], [1.0], ['str'], [{'a': 'b'}], [np.ones(3, )]])
@pytest.mark.parametrize('d', [{'a': 1}, {1: 2.0}, {'a': [1.0]}, {'a': np.ones(3, )}, {'a': {'a': 'b'}}])
@pytest.mark.parametrize('e', [Object()])
def test_change_env_kwargs(a: int, b: float, c: list, d: dict, e: Object):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'},
                            a=a, b=b, c=c, d=d, e=e
                            )
    assert a is env.a
    assert b is env.b
    assert c is env.c
    # Due to how gym works dict kwargs need to be copied and hence can only be checked to have the same content
    assert d == env.d
    assert e is env.e


@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
@pytest.mark.parametrize('add_time_aware_wrapper_before', [True, False])
def test_learn_sub_trajectories(env_wrap: Tuple[str, Type[RawInterfaceWrapper]], add_time_aware_wrapper_before: bool):
    env_id, wrapper_class = env_wrap
    env_step = TimeAwareObservation(fancy_gym.make(env_id, SEED))
    wrappers = [wrapper_class]

    # has time aware wrapper
    if add_time_aware_wrapper_before:
        wrappers += [TimeAwareObservation]

    env = fancy_gym.make_bb(env_id, [wrapper_class], {'learn_sub_trajectories': True},
                            {'trajectory_generator_type': 'promp'},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear'},
                            {'basis_generator_type': 'rbf'})

    assert env.learn_sub_trajectories
    assert env.traj_gen.learn_tau
    assert env.observation_space == env_step.observation_space

    env.reset()
    action = env.action_space.sample()
    obs, r, d, info = env.step(action)

    length = info['trajectory_length']

    factor = 1 / env.dt
    assert np.allclose(length * env.dt, np.round(factor * action[0]) / factor)
    assert np.allclose(length * env.dt, np.round(factor * env.traj_gen.tau.numpy()) / factor)
