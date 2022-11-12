from itertools import chain
from numbers import Number
from typing import Tuple, Type, Union, Optional, Callable, Iterable

import gym
import numpy as np
import pytest
import torch
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


@pytest.mark.parametrize('env_id', ENV_IDS)
def test_missing_wrapper(env_id: str):
    with pytest.raises(ValueError):
        fancy_gym.make_bb(env_id, [], {}, {}, {}, {}, {})


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
def test_missing_local_state(mp_type: str):
    env = fancy_gym.make_bb('toy-v0', [RawInterfaceWrapper], {},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    with pytest.raises(NotImplementedError):
        env.step(env.action_space.sample())


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
@pytest.mark.parametrize('verbose', [1, 2])
def test_verbosity(mp_type: str, env_wrap: Tuple[str, Type[RawInterfaceWrapper]], verbose: int):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {'verbose': verbose},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    info_keys = list(env.step(env.action_space.sample())[3].keys())

    env_step = fancy_gym.make(env_id, SEED)
    env_step.reset()
    info_keys_step = env_step.step(env_step.action_space.sample())[3].keys()

    assert all(e in info_keys for e in info_keys_step)
    assert 'trajectory_length' in info_keys

    if verbose >= 2:
        mp_keys = ['positions', 'velocities', 'step_actions', 'step_observations', 'step_rewards']
        assert all(e in info_keys for e in mp_keys)


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
def test_length(mp_type: str, env_wrap: Tuple[str, Type[RawInterfaceWrapper]]):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'})

    for _ in range(5):
        env.reset()
        length = env.step(env.action_space.sample())[3]['trajectory_length']

        assert length == env.spec.max_episode_steps


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('reward_aggregation', [np.sum, np.mean, np.median, lambda x: np.mean(x[::2])])
def test_aggregation(mp_type: str, reward_aggregation: Callable[[np.ndarray], float]):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'reward_aggregation': reward_aggregation},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'})
    env.reset()
    # ToyEnv only returns 1 as reward
    assert env.step(env.action_space.sample())[1] == reward_aggregation(np.ones(50, ))


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
def test_context_space(mp_type: str, env_wrap: Tuple[str, Type[RawInterfaceWrapper]]):
    env_id, wrapper_class = env_wrap
    env = fancy_gym.make_bb(env_id, [wrapper_class], {},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'})
    # check if observation space matches with the specified mask values which are true
    env_step = fancy_gym.make(env_id, SEED)
    wrapper = wrapper_class(env_step)
    assert env.observation_space.shape == wrapper.context_mask[wrapper.context_mask].shape


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('num_dof', [0, 1, 2, 5])
@pytest.mark.parametrize('num_basis', [0, 1, 2, 5])
@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
def test_action_space(mp_type: str, num_dof: int, num_basis: int, learn_tau: bool, learn_delay: bool):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {},
                            {'trajectory_generator_type': mp_type,
                             'action_dim': num_dof
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp',
                             'learn_tau': learn_tau,
                             'learn_delay': learn_delay
                             },
                            {'basis_generator_type': 'rbf',
                             'num_basis': num_basis
                             })

    base_dims = num_dof * num_basis
    additional_dims = num_dof if mp_type == 'dmp' else 0
    traj_modification_dims = int(learn_tau) + int(learn_delay)
    assert env.action_space.shape[0] == base_dims + traj_modification_dims + additional_dims


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [1.0])
@pytest.mark.parametrize('c', [[1], [1.0], ['str'], [{'a': 'b'}], [np.ones(3, )]])
@pytest.mark.parametrize('d', [{'a': 1}, {1: 2.0}, {'a': [1.0]}, {'a': np.ones(3, )}, {'a': {'a': 'b'}}])
@pytest.mark.parametrize('e', [Object()])
def test_change_env_kwargs(mp_type: str, a: int, b: float, c: list, d: dict, e: Object):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'},
                            a=a, b=b, c=c, d=d, e=e
                            )
    assert a is env.a
    assert b is env.b
    assert c is env.c
    # Due to how gym works dict kwargs need to be copied and hence can only be checked to have the same content
    assert d == env.d
    assert e is env.e


@pytest.mark.parametrize('mp_type', ['promp'])
@pytest.mark.parametrize('tau', [0.25, 0.5, 0.75, 1])
def test_learn_tau(mp_type: str, tau: float):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear',
                             'learn_tau': True,
                             'learn_delay': False
                             },
                            {'basis_generator_type': 'rbf',
                             }, seed=SEED)

    d = True
    for i in range(5):
        if d:
            env.reset()
        action = env.action_space.sample()
        action[0] = tau

        obs, r, d, info = env.step(action)

        length = info['trajectory_length']
        assert length == env.spec.max_episode_steps

        tau_time_steps = int(np.round(tau / env.dt))

        pos = info['positions'].flatten()
        vel = info['velocities'].flatten()

        # Check end is all same (only true for linear basis)
        assert np.all(pos[tau_time_steps:] == pos[-1])
        assert np.all(vel[tau_time_steps:] == vel[-1])

        # Check active trajectory section is different to end values
        assert np.all(pos[:tau_time_steps - 1] != pos[-1])
        assert np.all(vel[:tau_time_steps - 2] != vel[-1])


@pytest.mark.parametrize('mp_type', ['promp'])
@pytest.mark.parametrize('delay', [0, 0.25, 0.5, 0.75])
def test_learn_delay(mp_type: str, delay: float):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear',
                             'learn_tau': False,
                             'learn_delay': True
                             },
                            {'basis_generator_type': 'rbf',
                             }, seed=SEED)

    d = True
    for i in range(5):
        if d:
            env.reset()
        action = env.action_space.sample()
        action[0] = delay

        obs, r, d, info = env.step(action)

        length = info['trajectory_length']
        assert length == env.spec.max_episode_steps

        delay_time_steps = int(np.round(delay / env.dt))

        pos = info['positions'].flatten()
        vel = info['velocities'].flatten()

        # Check beginning is all same (only true for linear basis)
        assert np.all(pos[:max(1, delay_time_steps - 1)] == pos[0])
        assert np.all(vel[:max(1, delay_time_steps - 2)] == vel[0])

        # Check active trajectory section is different to beginning values
        assert np.all(pos[max(1, delay_time_steps):] != pos[0])
        assert np.all(vel[max(1, delay_time_steps)] != vel[0])


@pytest.mark.parametrize('mp_type', ['promp'])
@pytest.mark.parametrize('tau', [0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize('delay', [0.25, 0.5, 0.75, 1])
def test_learn_tau_and_delay(mp_type: str, tau: float, delay: float):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear',
                             'learn_tau': True,
                             'learn_delay': True
                             },
                            {'basis_generator_type': 'rbf',
                             }, seed=SEED)

    if env.spec.max_episode_steps * env.dt < delay + tau:
        return

    d = True
    for i in range(5):
        if d:
            env.reset()
        action = env.action_space.sample()
        action[0] = tau
        action[1] = delay

        obs, r, d, info = env.step(action)

        length = info['trajectory_length']
        assert length == env.spec.max_episode_steps

        tau_time_steps = int(np.round(tau / env.dt))
        delay_time_steps = int(np.round(delay / env.dt))
        joint_time_steps = delay_time_steps + tau_time_steps

        pos = info['positions'].flatten()
        vel = info['velocities'].flatten()

        # Check end is all same (only true for linear basis)
        assert np.all(pos[joint_time_steps:] == pos[-1])
        assert np.all(vel[joint_time_steps:] == vel[-1])

        # Check beginning is all same (only true for linear basis)
        assert np.all(pos[:delay_time_steps - 1] == pos[0])
        assert np.all(vel[:delay_time_steps - 2] == vel[0])

        # Check active trajectory section is different to beginning and end values
        active_pos = pos[delay_time_steps: joint_time_steps - 1]
        active_vel = vel[delay_time_steps: joint_time_steps - 2]
        assert np.all(active_pos != pos[-1]) and np.all(active_pos != pos[0])
        assert np.all(active_vel != vel[-1]) and np.all(active_vel != vel[0])


@pytest.mark.parametrize("action_constructor, expected_type", [(np.array, np.ndarray), (torch.tensor, torch.Tensor)])
@pytest.mark.parametrize('mp_type', ['promp'])
def test_get_trajectory(action_constructor: Callable[[Iterable[Number]], Union[np.ndarray, torch.Tensor]],
                        expected_type: Type,
                        mp_type: str
                        ):
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper], {'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'linear',
                             'learn_tau': False,
                             'learn_delay': True
                             },
                            {'basis_generator_type': 'rbf',
                             }, seed=SEED)
    action = action_constructor(env.action_space.sample())
    pos, vel = env.get_trajectory(action=action)

    assert isinstance(pos, expected_type)
    assert isinstance(vel, expected_type)
