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


@pytest.mark.parametrize('mp_type', ['promp', 'dmp'])
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

    env = fancy_gym.make_bb(env_id, [wrapper_class], {'replanning_schedule': replanning_schedule, 'verbose': 2},
                            {'trajectory_generator_type': mp_type},
                            {'controller_type': 'motor'},
                            {'phase_generator_type': 'exp'},
                            {'basis_generator_type': 'rbf'}, seed=SEED)

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
