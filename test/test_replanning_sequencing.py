from itertools import chain
from types import FunctionType
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


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64)
    dt = 0.02

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


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
@pytest.mark.parametrize('add_time_aware_wrapper_before', [True, False])
def test_learn_sub_trajectories(mp_type: str, env_wrap: Tuple[str, Type[RawInterfaceWrapper]],
                                add_time_aware_wrapper_before: bool):
    env_id, wrapper_class = env_wrap
    env_step = TimeAwareObservation(fancy_gym.make(env_id, SEED))
    wrappers = [wrapper_class]

    # has time aware wrapper
    if add_time_aware_wrapper_before:
        wrappers += [TimeAwareObservation]

    env = fancy_gym.make_bb(env_id, [wrapper_class], {'learn_sub_trajectories': True, 'verbose': 2},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'}, seed=SEED)

    assert env.learn_sub_trajectories
    assert env.traj_gen.learn_tau
    # This also verifies we are not adding the TimeAwareObservationWrapper twice
    assert env.observation_space == env_step.observation_space

    d = True

    for i in range(25):
        if d:
            env.reset()
        action = env.action_space.sample()
        obs, r, d, info = env.step(action)

        length = info['trajectory_length']

        if not d:
            assert length == np.round(action[0] / env.dt)
            assert length == np.round(env.traj_gen.tau.numpy() / env.dt)
        else:
            # When done trajectory could be shorter due to termination.
            assert length <= np.round(action[0] / env.dt)
            assert length <= np.round(env.traj_gen.tau.numpy() / env.dt)


@pytest.mark.parametrize('mp_type', ['promp', 'dmp', 'prodmp'])
@pytest.mark.parametrize('env_wrap', zip(ENV_IDS, WRAPPERS))
@pytest.mark.parametrize('add_time_aware_wrapper_before', [True, False])
@pytest.mark.parametrize('replanning_time', [10, 100, 1000])
def test_replanning_time(mp_type: str, env_wrap: Tuple[str, Type[RawInterfaceWrapper]],
                         add_time_aware_wrapper_before: bool, replanning_time: int):
    env_id, wrapper_class = env_wrap
    env_step = TimeAwareObservation(fancy_gym.make(env_id, SEED))
    wrappers = [wrapper_class]

    # has time aware wrapper
    if add_time_aware_wrapper_before:
        wrappers += [TimeAwareObservation]

    replanning_schedule = lambda c_pos, c_vel, obs, c_action, t: t % replanning_time == 0

    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if 'dmp' in mp_type else 'linear'

    env = fancy_gym.make_bb(env_id, [wrapper_class], {'replanning_schedule': replanning_schedule, 'verbose': 2},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': phase_generator_type},
                            {'basis_generator_type': basis_generator_type}, seed=SEED)

    assert env.do_replanning
    assert callable(env.replanning_schedule)
    # This also verifies we are not adding the TimeAwareObservationWrapper twice
    assert env.observation_space == env_step.observation_space

    env.reset()

    episode_steps = env_step.spec.max_episode_steps // replanning_time
    # Make 3 episodes, total steps depend on the replanning steps
    for i in range(3 * episode_steps):
        action = env.action_space.sample()
        obs, r, d, info = env.step(action)

        length = info['trajectory_length']

        if d:
            # Check if number of steps until termination match the replanning interval
            print(d, (i + 1), episode_steps)
            assert (i + 1) % episode_steps == 0
            env.reset()

        assert replanning_schedule(None, None, None, None, length)

@pytest.mark.parametrize('mp_type', ['promp', 'prodmp'])
@pytest.mark.parametrize('max_planning_times', [1, 2, 3, 4])
@pytest.mark.parametrize('sub_segment_steps', [5, 10])
def test_max_planning_times(mp_type: str, max_planning_times: int, sub_segment_steps: int):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if mp_type == 'prodmp' else 'linear'
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper],
                            {'max_planning_times': max_planning_times,
                             'replanning_schedule': lambda pos, vel, obs, action, t: t % sub_segment_steps == 0,
                             'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': phase_generator_type,
                             'learn_tau': False,
                             'learn_delay': False
                             },
                            {'basis_generator_type': basis_generator_type,
                             },
                            seed=SEED)
    _ = env.reset()
    d = False
    planning_times = 0
    while not d:
        _, _, d, _ = env.step(env.action_space.sample())
        planning_times += 1
    assert planning_times == max_planning_times

@pytest.mark.parametrize('mp_type', ['promp', 'prodmp'])
@pytest.mark.parametrize('max_planning_times', [1, 2, 3, 4])
@pytest.mark.parametrize('sub_segment_steps', [5, 10])
@pytest.mark.parametrize('tau', [0.5, 1.0, 1.5, 2.0])
def test_replanning_with_learn_tau(mp_type: str, max_planning_times: int, sub_segment_steps: int, tau: float):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if mp_type == 'prodmp' else 'linear'
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper],
                            {'replanning_schedule': lambda pos, vel, obs, action, t: t % sub_segment_steps == 0,
                             'max_planning_times': max_planning_times,
                             'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': phase_generator_type,
                             'learn_tau': True,
                             'learn_delay': False
                             },
                            {'basis_generator_type': basis_generator_type,
                             },
                            seed=SEED)
    _ = env.reset()
    d = False
    planning_times = 0
    while not d:
        action = env.action_space.sample()
        action[0] = tau
        _, _, d, info = env.step(action)
        planning_times += 1
    assert planning_times == max_planning_times

@pytest.mark.parametrize('mp_type', ['promp', 'prodmp'])
@pytest.mark.parametrize('max_planning_times', [1, 2, 3, 4])
@pytest.mark.parametrize('sub_segment_steps', [5, 10])
@pytest.mark.parametrize('delay', [0.1, 0.25, 0.5, 0.75])
def test_replanning_with_learn_delay(mp_type: str, max_planning_times: int, sub_segment_steps: int, delay: float):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if mp_type == 'prodmp' else 'linear'
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper],
                        {'replanning_schedule': lambda pos, vel, obs, action, t: t % sub_segment_steps == 0,
                         'max_planning_times': max_planning_times,
                         'verbose': 2},
                        {'trajectory_generator_type': mp_type,
                         },
                        {'controller_type': 'motor'},
                        {'phase_generator_type': phase_generator_type,
                         'learn_tau': False,
                         'learn_delay': True
                         },
                        {'basis_generator_type': basis_generator_type,
                         },
                        seed=SEED)
    _ = env.reset()
    d = False
    planning_times = 0
    while not d:
        action = env.action_space.sample()
        action[0] = delay
        _, _, d, info = env.step(action)

        delay_time_steps = int(np.round(delay / env.dt))
        pos = info['positions'].flatten()
        vel = info['velocities'].flatten()

        # Check beginning is all same (only true for linear basis)
        if planning_times == 0:
            assert np.all(pos[:max(1, delay_time_steps - 1)] == pos[0])
            assert np.all(vel[:max(1, delay_time_steps - 2)] == vel[0])

        # only valid when delay < sub_segment_steps
        elif planning_times > 0 and delay_time_steps < sub_segment_steps:
            assert np.all(pos[1:max(1, delay_time_steps - 1)] != pos[0])
            assert np.all(vel[1:max(1, delay_time_steps - 2)] != vel[0])

        # Check active trajectory section is different to beginning values
        assert np.all(pos[max(1, delay_time_steps):] != pos[0])
        assert np.all(vel[max(1, delay_time_steps)] != vel[0])

        planning_times += 1

    assert planning_times == max_planning_times

@pytest.mark.parametrize('mp_type', ['promp', 'prodmp'])
@pytest.mark.parametrize('max_planning_times', [1, 2, 3])
@pytest.mark.parametrize('sub_segment_steps', [5, 10, 15])
@pytest.mark.parametrize('delay', [0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('tau', [0.5, 0.75, 1.0])
def test_replanning_with_learn_delay_and_tau(mp_type: str, max_planning_times: int, sub_segment_steps: int,
                                             delay: float, tau: float):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if mp_type == 'prodmp' else 'linear'
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper],
                        {'replanning_schedule': lambda pos, vel, obs, action, t: t % sub_segment_steps == 0,
                         'max_planning_times': max_planning_times,
                         'verbose': 2},
                        {'trajectory_generator_type': mp_type,
                         },
                        {'controller_type': 'motor'},
                        {'phase_generator_type': phase_generator_type,
                         'learn_tau': True,
                         'learn_delay': True
                         },
                        {'basis_generator_type': basis_generator_type,
                         },
                        seed=SEED)
    _ = env.reset()
    d = False
    planning_times = 0
    while not d:
        action = env.action_space.sample()
        action[0] = tau
        action[1] = delay
        _, _, d, info = env.step(action)

        delay_time_steps = int(np.round(delay / env.dt))

        pos = info['positions'].flatten()
        vel = info['velocities'].flatten()

        # Delay only applies to first planning time
        if planning_times == 0:
            # Check delay is applied
            assert np.all(pos[:max(1, delay_time_steps - 1)] == pos[0])
            assert np.all(vel[:max(1, delay_time_steps - 2)] == vel[0])
            # Check active trajectory section is different to beginning values
            assert np.all(pos[max(1, delay_time_steps):] != pos[0])
            assert np.all(vel[max(1, delay_time_steps)] != vel[0])

        planning_times += 1

    assert planning_times == max_planning_times

@pytest.mark.parametrize('mp_type', ['promp', 'prodmp'])
@pytest.mark.parametrize('max_planning_times', [1, 2, 3, 4])
@pytest.mark.parametrize('sub_segment_steps', [5, 10])
def test_replanning_schedule(mp_type: str, max_planning_times: int, sub_segment_steps: int):
    basis_generator_type = 'prodmp' if mp_type == 'prodmp' else 'rbf'
    phase_generator_type = 'exp' if mp_type == 'prodmp' else 'linear'
    env = fancy_gym.make_bb('toy-v0', [ToyWrapper],
                            {'max_planning_times': max_planning_times,
                             'replanning_schedule': lambda pos, vel, obs, action, t: t % sub_segment_steps == 0,
                             'verbose': 2},
                            {'trajectory_generator_type': mp_type,
                             },
                            {'controller_type': 'motor'},
                            {'phase_generator_type': phase_generator_type,
                             'learn_tau': False,
                             'learn_delay': False
                             },
                            {'basis_generator_type': basis_generator_type,
                             },
                            seed=SEED)
    _ = env.reset()
    d = False
    for i in range(max_planning_times):
        _, _, d, _ = env.step(env.action_space.sample())
    assert d
