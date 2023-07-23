from fancy_gym.utils.wrappers import TimeAwareObservation
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator
from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.controller_factory import get_controller
from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.black_box_wrapper import BlackBoxWrapper
import uuid
from collections.abc import MutableMapping
from math import ceil
from typing import Iterable, Type, Union, Optional

import gymnasium as gym
from gymnasium import make
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import TimeLimit

from fancy_gym.utils.env_compatibility import EnvCompatibility
from fancy_gym.utils.wrappers import FlattenObservation

try:
    import shimmy
    from shimmy.dm_control_compatibility import EnvType
except ImportError:
    pass

try:
    import metaworld
except Exception:
    # catch Exception as Import error does not catch missing mujoco-py
    pass


def _make_wrapped_env(env: gym.Env, wrappers: Iterable[Type[gym.Wrapper]], seed=1, fallback_max_steps=None):
    """
    Helper function for creating a wrapped gym environment using MPs.
    It adds all provided wrappers to the specified environment and verifies at least one RawInterfaceWrapper is
    provided to expose the interface for MPs.

    Args:
        env: base environemnt to wrap
        wrappers: list of wrappers (at least an RawInterfaceWrapper),
        seed: seed of environment

    Returns: gym environment with all specified wrappers applied

    """
    if fallback_max_steps:
        env = ensure_finite_time(env, fallback_max_steps)
    has_black_box_wrapper = False
    head = env
    while hasattr(head, 'env'):
        if isinstance(head, RawInterfaceWrapper):
            has_black_box_wrapper = True
            break
        head = head.env
    for w in wrappers:
        # only wrap the environment if not BlackBoxWrapper, e.g. for vision
        if issubclass(w, RawInterfaceWrapper):
            has_black_box_wrapper = True
        env = w(env)
    if not has_black_box_wrapper:
        raise ValueError("A RawInterfaceWrapper is required in order to leverage movement primitive environments.")
    return env


def make_bb(
        env: Union[gym.Env, str], wrappers: Iterable, black_box_kwargs: MutableMapping, traj_gen_kwargs: MutableMapping,
        controller_kwargs: MutableMapping, phase_kwargs: MutableMapping, basis_kwargs: MutableMapping,
        time_limit: int = None, fallback_max_steps: int = None, **kwargs):
    """
    This can also be used standalone for manually building a custom DMP environment.
    Args:
        black_box_kwargs: kwargs for the black-box wrapper
        basis_kwargs: kwargs for the basis generator
        phase_kwargs: kwargs for the phase generator
        controller_kwargs: kwargs for the tracking controller
        env: step based environment (or environment id),
        wrappers: list of wrappers (at least an RawInterfaceWrapper),
        seed: seed of environment
        traj_gen_kwargs: dict of at least {num_dof: int, num_basis: int} for DMP

    Returns: DMP wrapped gym env

    """
    _verify_time_limit(traj_gen_kwargs.get("duration"), time_limit)

    learn_sub_trajs = black_box_kwargs.get('learn_sub_trajectories')
    do_replanning = black_box_kwargs.get('replanning_schedule')
    if learn_sub_trajs and do_replanning:
        raise ValueError('Cannot used sub-trajectory learning and replanning together.')

    # add time_step observation when replanning
    if (learn_sub_trajs or do_replanning) and not any(issubclass(w, TimeAwareObservation) for w in wrappers):
        # Add as first wrapper in order to alter observation
        wrappers.insert(0, TimeAwareObservation)

    if isinstance(env, str):
        env = make(env, **kwargs)

    env = _make_wrapped_env(env=env, wrappers=wrappers, fallback_max_steps=fallback_max_steps)

    # BB expects a spaces.Box to be exposed, need to convert for dict-observations
    if type(env.observation_space) == gym.spaces.dict.Dict:
        env = FlattenObservation(env)

    traj_gen_kwargs['action_dim'] = traj_gen_kwargs.get('action_dim', np.prod(env.action_space.shape).item())

    if black_box_kwargs.get('duration') is None:
        black_box_kwargs['duration'] = get_env_duration(env)
    if phase_kwargs.get('tau') is None:
        phase_kwargs['tau'] = black_box_kwargs['duration']

    if learn_sub_trajs is not None:
        # We have to learn the length when learning sub_trajectories trajectories
        phase_kwargs['learn_tau'] = True

    # set tau bounds to minimum of two env steps otherwise computing the velocity is not possible.
    # maximum is full duration of one episode.
    if phase_kwargs.get('learn_tau') and phase_kwargs.get('tau_bound') is None:
        phase_kwargs["tau_bound"] = [env.dt * 2, black_box_kwargs['duration']]

    # Max delay is full duration minus two steps due to above reason
    if phase_kwargs.get('learn_delay') and phase_kwargs.get('delay_bound') is None:
        phase_kwargs["delay_bound"] = [0, black_box_kwargs['duration'] - env.dt * 2]

    phase_gen = get_phase_generator(**phase_kwargs)
    basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_kwargs)
    controller = get_controller(**controller_kwargs)
    traj_gen = get_trajectory_generator(basis_generator=basis_gen, **traj_gen_kwargs)

    bb_env = BlackBoxWrapper(env, trajectory_generator=traj_gen, tracking_controller=controller,
                             **black_box_kwargs)

    return bb_env


def ensure_finite_time(env: gym.Env, fallback_max_steps=500):
    cur_limit = env.spec.max_episode_steps
    if not cur_limit:
        if hasattr(env.unwrapped, 'max_path_length'):
            return TimeLimit(env, env.unwrapped.__getattribute__('max_path_length'))
        return TimeLimit(env, fallback_max_steps)
    return env


def get_env_duration(env: gym.Env):
    try:
        duration = env.spec.max_episode_steps * env.dt
    except (AttributeError, TypeError) as e:
        if env.env_type is EnvType.COMPOSER:
            max_episode_steps = ceil(env.unwrapped._time_limit / env.dt)
        elif env.env_type is EnvType.RL_CONTROL:
            max_episode_steps = int(env.unwrapped._step_limit)
        else:
            raise e
        duration = max_episode_steps * env.control_timestep()
    return duration


def _verify_time_limit(mp_time_limit: Union[None, float], env_time_limit: Union[None, float]):
    """
    When using DMC check if a manually specified time limit matches the trajectory duration the MP receives.
    Mostly, the time_limit for DMC is not specified and the default values from DMC are taken.
    This check, however, can only been done after instantiating the environment.
    It can be found in the BaseMP class.

    Args:
        mp_time_limit: max trajectory length of traj_gen in seconds
        env_time_limit: max trajectory length of DMC environment in seconds

    Returns:

    """
    if mp_time_limit is not None and env_time_limit is not None:
        assert mp_time_limit == env_time_limit, \
            f"The specified 'time_limit' of {env_time_limit}s does not match " \
            f"the duration of {mp_time_limit}s for the MP."


def _verify_dof(base_env: gym.Env, dof: int):
    action_shape = np.prod(base_env.action_space.shape)
    assert dof == action_shape, \
        f"The specified degrees of freedom ('num_dof') {dof} do not match " \
        f"the action space of {action_shape} the base environments"
