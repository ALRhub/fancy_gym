import warnings
from copy import deepcopy
from typing import Iterable, Type, Union, MutableMapping

import gym
import numpy as np
from gym.envs.registration import EnvSpec, registry
from gym.wrappers import TimeAwareObservation

from alr_envs.black_box.black_box_wrapper import BlackBoxWrapper
from alr_envs.black_box.controller.controller_factory import get_controller
from alr_envs.black_box.factory.basis_generator_factory import get_basis_generator
from alr_envs.black_box.factory.phase_generator_factory import get_phase_generator
from alr_envs.black_box.factory.trajectory_generator_factory import get_trajectory_generator
from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper
from alr_envs.utils.utils import nested_update


def make_rank(env_id: str, seed: int, rank: int = 0, return_callable=True, **kwargs):
    """
    TODO: Do we need this?
    Generate a callable to create a new gym environment with a given seed.
    The rank is added to the seed and can be used for example when using vector environments.
    E.g. [make_rank("my_env_name-v0", 123, i) for i in range(8)] creates a list of 8 environments
    with seeds 123 through 130.
    Hence, testing environments should be seeded with a value which is offset by the number of training environments.
    Here e.g. [make_rank("my_env_name-v0", 123 + 8, i) for i in range(5)] for 5 testing environmetns

    Args:
        env_id: name of the environment
        seed: seed for deterministic behaviour
        rank: environment rank for deterministic over multiple seeds behaviour
        return_callable: If True returns a callable to create the environment instead of the environment itself.

    Returns:

    """

    def f():
        return make(env_id, seed + rank, **kwargs)

    return f if return_callable else f()


def make(env_id, seed, **kwargs):
    # TODO: This doesn't work with gym ==0.21.0
    # This access is required to allow for nested dict updates
    spec = registry.get(env_id)
    all_kwargs = deepcopy(spec.kwargs)
    nested_update(all_kwargs, kwargs)
    return _make(env_id, seed, **all_kwargs)


def _make(env_id: str, seed, **kwargs):
    """
    Converts an env_id to an environment with the gym API.
    This also works for DeepMind Control Suite interface_wrappers
    for which domain name and task name are expected to be separated by "-".
    Args:
        env_id: gym name or env_id of the form "domain_name-task_name" for DMC tasks
        **kwargs: Additional kwargs for the constructor such as pixel observations, etc.

    Returns: Gym environment

    """
    if any(deprec in env_id for deprec in ["DetPMP", "detpmp"]):
        warnings.warn("DetPMP is deprecated and converted to ProMP")
        env_id = env_id.replace("DetPMP", "ProMP")
        env_id = env_id.replace("detpmp", "promp")

    try:
        # Add seed to kwargs in case it is a predefined gym+dmc hybrid environment.
        if env_id.startswith("dmc"):
            kwargs.update({"seed": seed})

        # Gym
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except gym.error.Error:

        # MetaWorld env
        import metaworld
        if env_id in metaworld.ML1.ENV_NAMES:
            env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id + "-goal-observable"](seed=seed, **kwargs)
            # setting this avoids generating the same initialization after each reset
            env._freeze_rand_vec = False
            # Manually set spec, as metaworld environments are not registered via gym
            env.unwrapped.spec = EnvSpec(env_id)
            # Set Timelimit based on the maximum allowed path length of the environment
            env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env.goal_space.seed(seed)

        else:
            # DMC
            from alr_envs import make_dmc
            env = make_dmc(env_id, seed=seed, **kwargs)

            if not env.base_step_limit == env.spec.max_episode_steps:
                raise ValueError(f"The specified 'episode_length' of {env.spec.max_episode_steps} steps for gym "
                                 f"is different from the DMC environment specification of {env.base_step_limit} steps.")

    return env


def _make_wrapped_env(
        env_id: str, wrappers: Iterable[Type[gym.Wrapper]], seed=1, **kwargs
):
    """
    Helper function for creating a wrapped gym environment using MPs.
    It adds all provided wrappers to the specified environment and verifies at least one RawInterfaceWrapper is
    provided to expose the interface for MPs.

    Args:
        env_id: name of the environment
        wrappers: list of wrappers (at least an RawInterfaceWrapper),
        seed: seed of environment

    Returns: gym environment with all specified wrappers applied

    """
    # _env = gym.make(env_id)
    _env = make(env_id, seed, **kwargs)
    has_black_box_wrapper = False
    for w in wrappers:
        # only wrap the environment if not BlackBoxWrapper, e.g. for vision
        if issubclass(w, RawInterfaceWrapper):
            has_black_box_wrapper = True
        _env = w(_env)
    if not has_black_box_wrapper:
        raise ValueError("An RawInterfaceWrapper is required in order to leverage movement primitive environments.")
    return _env


def make_bb(
        env_id: str, wrappers: Iterable, black_box_kwargs: MutableMapping, traj_gen_kwargs: MutableMapping,
        controller_kwargs: MutableMapping, phase_kwargs: MutableMapping, basis_kwargs: MutableMapping, seed=1,
        **kwargs):
    """
    This can also be used standalone for manually building a custom DMP environment.
    Args:
        black_box_kwargs: kwargs for the black-box wrapper
        basis_kwargs: kwargs for the basis generator
        phase_kwargs: kwargs for the phase generator
        controller_kwargs: kwargs for the tracking controller
        env_id: base_env_name,
        wrappers: list of wrappers (at least an RawInterfaceWrapper),
        seed: seed of environment
        traj_gen_kwargs: dict of at least {num_dof: int, num_basis: int} for DMP

    Returns: DMP wrapped gym env

    """
    _verify_time_limit(traj_gen_kwargs.get("duration", None), kwargs.get("time_limit", None))
    _env = _make_wrapped_env(env_id=env_id, wrappers=wrappers, seed=seed, **kwargs)

    learn_sub_trajs = black_box_kwargs.get('learn_sub_trajectories')
    do_replanning = black_box_kwargs.get('replanning_schedule')
    if learn_sub_trajs and do_replanning:
        raise ValueError('Cannot used sub-trajectory learning and replanning together.')

    if learn_sub_trajs or do_replanning:
        # add time_step observation when replanning
        kwargs['wrappers'].append(TimeAwareObservation)

    traj_gen_kwargs['action_dim'] = traj_gen_kwargs.get('action_dim', np.prod(_env.action_space.shape).item())

    if black_box_kwargs.get('duration') is None:
        black_box_kwargs['duration'] = _env.spec.max_episode_steps * _env.dt
    if phase_kwargs.get('tau') is None:
        phase_kwargs['tau'] = black_box_kwargs['duration']

    if learn_sub_trajs is not None:
        # We have to learn the length when learning sub_trajectories trajectories
        phase_kwargs['learn_tau'] = True

    phase_gen = get_phase_generator(**phase_kwargs)
    basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_kwargs)
    controller = get_controller(**controller_kwargs)
    traj_gen = get_trajectory_generator(basis_generator=basis_gen, **traj_gen_kwargs)

    bb_env = BlackBoxWrapper(_env, trajectory_generator=traj_gen, tracking_controller=controller,
                             **black_box_kwargs)

    return bb_env


def make_bb_env_helper(**kwargs):
    """
    Helper function for registering a black box gym environment.
    Args:
        **kwargs: expects at least the following:
        {
        "name": base environment name.
        "wrappers": list of wrappers (at least an BlackBoxWrapper is required),
        "traj_gen_kwargs": {
            "trajectory_generator_type": type_of_your_movement_primitive,
            non default arguments for the movement primitive instance
            ...
            }
        "controller_kwargs": {
            "controller_type": type_of_your_controller,
            non default arguments for the tracking_controller instance
            ...
            },
        "basis_generator_kwargs": {
            "basis_generator_type": type_of_your_basis_generator,
            non default arguments for the basis generator instance
            ...
            },
        "phase_generator_kwargs": {
            "phase_generator_type": type_of_your_phase_generator,
            non default arguments for the phase generator instance
            ...
            },
        }

    Returns: MP wrapped gym env

    """
    seed = kwargs.pop("seed", None)
    wrappers = kwargs.pop("wrappers")

    traj_gen_kwargs = kwargs.pop("trajectory_generator_kwargs", {})
    black_box_kwargs = kwargs.pop('black_box_kwargs', {})
    contr_kwargs = kwargs.pop("controller_kwargs", {})
    phase_kwargs = kwargs.pop("phase_generator_kwargs", {})
    basis_kwargs = kwargs.pop("basis_generator_kwargs", {})

    return make_bb(env_id=kwargs.pop("name"), wrappers=wrappers,
                   black_box_kwargs=black_box_kwargs,
                   traj_gen_kwargs=traj_gen_kwargs, controller_kwargs=contr_kwargs,
                   phase_kwargs=phase_kwargs,
                   basis_kwargs=basis_kwargs, **kwargs, seed=seed)


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
