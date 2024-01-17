import logging
import re
import uuid
from collections.abc import MutableMapping
from copy import deepcopy
from math import ceil
from typing import Iterable, Type, Union

import gym
import numpy as np
from gym.envs.registration import register, registry

try:
    from dm_control import suite, manipulation
except ImportError:
    pass

try:
    import metaworld
except Exception:
    # catch Exception as Import error does not catch missing mujoco-py
    pass

import fancy_gym
from fancy_gym.black_box.black_box_wrapper import BlackBoxWrapper
from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.factory.controller_factory import get_controller
from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.time_aware_observation import TimeAwareObservation
from fancy_gym.utils.utils import nested_update


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


def make(env_id: str, seed: int, **kwargs):
    """
    Converts an env_id to an environment with the gym API.
    This also works for DeepMind Control Suite environments that are wrapped using the DMCWrapper, they can be
    specified with "dmc:domain_name-task_name"
    Analogously, metaworld tasks can be created as "metaworld:env_id-v2".

    Args:
        env_id: spec or env_id for gym tasks, external environments require a domain specification
        **kwargs: Additional kwargs for the constructor such as pixel observations, etc.

    Returns: Gym environment

    """

    if ':' in env_id:
        split_id = env_id.split(':')
        framework, env_id = split_id[-2:]
    else:
        framework = None

    if framework == 'metaworld':
        # MetaWorld environment
        env = make_metaworld(env_id, seed, **kwargs)
    elif framework == 'dmc':
        # DeepMind Control environment
        env = make_dmc(env_id, seed, **kwargs)
    else:
        env = make_gym(env_id, seed, **kwargs)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def _make_wrapped_env(env_id: str, wrappers: Iterable[Type[gym.Wrapper]], seed=1, **kwargs):
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
        _env = w(_env, **kwargs)
    if not has_black_box_wrapper:
        raise ValueError("A RawInterfaceWrapper is required in order to leverage movement primitive environments.")
    return _env


def make_bb(
        env_id: str, wrappers: Iterable, black_box_kwargs: MutableMapping, traj_gen_kwargs: MutableMapping,
        controller_kwargs: MutableMapping, phase_kwargs: MutableMapping, basis_kwargs: MutableMapping, seed: int = 1,
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
    _verify_time_limit(traj_gen_kwargs.get("duration"), kwargs.get("time_limit"))

    learn_sub_trajs = black_box_kwargs.get('learn_sub_trajectories')
    do_replanning = black_box_kwargs.get('replanning_schedule')
    if learn_sub_trajs and do_replanning:
        raise ValueError('Cannot used sub-trajectory learning and replanning together.')

    # add time_step observation when replanning
    if (learn_sub_trajs or do_replanning) and not any(issubclass(w, TimeAwareObservation) for w in wrappers):
        # Add as first wrapper in order to alter observation
        wrappers.insert(0, TimeAwareObservation)

    env = _make_wrapped_env(env_id=env_id, wrappers=wrappers, seed=seed, **kwargs)

    traj_gen_kwargs['action_dim'] = traj_gen_kwargs.get('action_dim', np.prod(env.action_space.shape).item())

    if black_box_kwargs.get('duration') is None:
        black_box_kwargs['duration'] = env.spec.max_episode_steps * env.dt
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
    # seed = kwargs.get("seed", None)
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


def make_dmc(
        env_id: str,
        seed: int = None,
        visualize_reward: bool = True,
        time_limit: Union[None, float] = None,
        **kwargs
):
    if not re.match(r"\w+-\w+", env_id):
        raise ValueError("env_id does not have the following structure: 'domain_name-task_name'")
    domain_name, task_name = env_id.split("-")

    if task_name.endswith("_vision"):
        # TODO
        raise ValueError("The vision interface for manipulation tasks is currently not supported.")

    if (domain_name, task_name) not in suite.ALL_TASKS and task_name not in manipulation.ALL:
        raise ValueError(f'Specified domain "{domain_name}" and task "{task_name}" combination does not exist.')

    # env_id = f'dmc_{domain_name}_{task_name}_{seed}-v1'
    gym_id = uuid.uuid4().hex + '-v1'

    task_kwargs = {'random': seed}
    if time_limit is not None:
        task_kwargs['time_limit'] = time_limit

    # create task
    # Accessing private attribute because DMC does not expose time_limit or step_limit.
    # Only the current time_step/time as well as the control_timestep can be accessed.
    if domain_name == "manipulation":
        env = manipulation.load(environment_name=task_name, seed=seed)
        max_episode_steps = ceil(env._time_limit / env.control_timestep())
    else:
        env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                         visualize_reward=visualize_reward, environment_kwargs=kwargs)
        max_episode_steps = int(env._step_limit)

    register(
        id=gym_id,
        entry_point='fancy_gym.dmc.dmc_wrapper:DMCWrapper',
        kwargs={'env': lambda: env},
        max_episode_steps=max_episode_steps,
    )

    env = gym.make(gym_id)
    env.seed(seed)
    return env


def make_metaworld(env_id: str, seed: int, **kwargs):
    if env_id not in metaworld.ML1.ENV_NAMES:
        raise ValueError(f'Specified environment "{env_id}" not present in metaworld ML1.')

    _env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id + "-goal-observable"](seed=seed, **kwargs)

    # setting this avoids generating the same initialization after each reset
    _env._freeze_rand_vec = False
    # New argument to use global seeding
    _env.seeded_rand_vec = True

    gym_id = uuid.uuid4().hex + '-v1'

    register(
        id=gym_id,
        entry_point=lambda: _env,
        max_episode_steps=_env.max_path_length,
    )

    # TODO enable checker when the incorrect dtype of obs and observation space are fixed by metaworld
    env = gym.make(gym_id, disable_env_checker=True)
    return env


def make_gym(env_id, seed, **kwargs):
    """
    Create
    Args:
        env_id:
        seed:
        **kwargs:

    Returns:

    """
    # Getting the existing keywords to allow for nested dict updates for BB envs
    # gym only allows for non nested updates.
    try:
        all_kwargs = deepcopy(registry.get(env_id).kwargs)
    except AttributeError as e:
        logging.error(f'The gym environment with id {env_id} could not been found.')
        raise e
    nested_update(all_kwargs, kwargs)
    kwargs = all_kwargs

    # Add seed to kwargs for bb environments to pass seed to step environments
    all_bb_envs = sum(fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values(), [])
    if env_id in all_bb_envs:
        kwargs.update({"seed": seed})

    # Gym
    env = gym.make(env_id, **kwargs)
    return env


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
