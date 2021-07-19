import logging
from typing import Iterable, List, Type, Union

import gym

from mp_env_api.interface_wrappers.mp_env_wrapper import MPEnvWrapper
from mp_env_api.mp_wrappers.detpmp_wrapper import DetPMPWrapper
from mp_env_api.mp_wrappers.dmp_wrapper import DmpWrapper


def make_env_rank(env_id: str, seed: int, rank: int = 0, **kwargs):
    """
    TODO: Do we need this?
    Generate a callable to create a new gym environment with a given seed.
    The rank is added to the seed and can be used for example when using vector environments.
    E.g. [make_env_rank("my_env_name-v0", 123, i) for i in range(8)] creates a list of 8 environments
    with seeds 123 through 130.
    Hence, testing environments should be seeded with a value which is offset by the number of training environments.
    Here e.g. [make_env_rank("my_env_name-v0", 123 + 8, i) for i in range(5)] for 5 testing environmetns

    Args:
        env_id: name of the environment
        seed: seed for deterministic behaviour
        rank: environment rank for deterministic over multiple seeds behaviour

    Returns:

    """
    return lambda: make_env(env_id, seed + rank, **kwargs)


def make_env(env_id: str, seed, **kwargs):
    """
    Converts an env_id to an environment with the gym API.
    This also works for DeepMind Control Suite interface_wrappers
    for which domain name and task name are expected to be separated by "-".
    Args:
        env_id: gym name or env_id of the form "domain_name-task_name" for DMC tasks
        **kwargs: Additional kwargs for the constructor such as pixel observations, etc.

    Returns: Gym environment

    """
    try:
        # Add seed to kwargs in case it is a predefined gym+dmc hybrid environment.
        if env_id.startswith("dmc"):
            kwargs.update({"seed": seed})

        # Gym
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
    except gym.error.Error:
        # DMC
        from alr_envs.utils import make
        env = make(env_id, seed=seed, **kwargs)

        assert env.base_step_limit == env.spec.max_episode_steps, \
            f"The specified 'episode_length' of {env.spec.max_episode_steps} steps for gym is different from " \
            f"the DMC environment specification of {env.base_step_limit} steps."

    return env


def _make_wrapped_env(env_id: str, wrappers: Iterable[Type[gym.Wrapper]], seed=1, **kwargs):
    """
    Helper function for creating a wrapped gym environment using MPs.
    It adds all provided wrappers to the specified environment and verifies at least one MPEnvWrapper is
    provided to expose the interface for MPs.

    Args:
        env_id: name of the environment
        wrappers: list of wrappers (at least an MPEnvWrapper),
        seed: seed of environment

    Returns: gym environment with all specified wrappers applied

    """
    # _env = gym.make(env_id)
    _env = make_env(env_id, seed, **kwargs)

    assert any(issubclass(w, MPEnvWrapper) for w in wrappers), \
        "At least an MPEnvWrapper is required in order to leverage motion primitive environments."
    for w in wrappers:
        _env = w(_env)

    return _env


def make_dmp_env(env_id: str, wrappers: Iterable, seed=1, mp_kwargs={}, **kwargs):
    """
    This can also be used standalone for manually building a custom DMP environment.
    Args:
        env_id: base_env_name,
        wrappers: list of wrappers (at least an MPEnvWrapper),
        seed: seed of environment
        mp_kwargs: dict of at least {num_dof: int, num_basis: int} for DMP

    Returns: DMP wrapped gym env

    """
    verify_time_limit(mp_kwargs.get("duration", None), kwargs.get("time_limit", None))

    _env = _make_wrapped_env(env_id=env_id, wrappers=wrappers, seed=seed, **kwargs)
    return DmpWrapper(_env, **mp_kwargs)


def make_detpmp_env(env_id: str, wrappers: Iterable, seed=1, mp_kwargs={}, **kwargs):
    """
    This can also be used standalone for manually building a custom Det ProMP environment.
    Args:
        env_id: base_env_name,
        wrappers: list of wrappers (at least an MPEnvWrapper),
        mp_kwargs: dict of at least {num_dof: int, num_basis: int, width: int}

    Returns: DMP wrapped gym env

    """
    verify_time_limit(mp_kwargs.get("duration", None), kwargs.get("time_limit", None))

    _env = _make_wrapped_env(env_id=env_id, wrappers=wrappers, seed=seed, **kwargs)
    return DetPMPWrapper(_env, **mp_kwargs)


def make_dmp_env_helper(**kwargs):
    """
    Helper function for registering a DMP gym environments.
    Args:
        **kwargs: expects at least the following:
        {
        "name": base_env_name,
        "wrappers": list of wrappers (at least an MPEnvWrapper),
        "mp_kwargs": dict of at least {num_dof: int, num_basis: int} for DMP
        }

    Returns: DMP wrapped gym env

    """
    seed = kwargs.pop("seed", None)
    return make_dmp_env(env_id=kwargs.pop("name"), wrappers=kwargs.pop("wrappers"), seed=seed,
                        mp_kwargs=kwargs.pop("mp_kwargs"), **kwargs)


def make_detpmp_env_helper(**kwargs):
    """
    Helper function for registering ProMP gym environments.
    This can also be used standalone for manually building a custom ProMP environment.
    Args:
        **kwargs: expects at least the following:
        {
        "name": base_env_name,
        "wrappers": list of wrappers (at least an MPEnvWrapper),
        "mp_kwargs": dict of at least {num_dof: int, num_basis: int, width: int}
        }

    Returns: DMP wrapped gym env

    """
    seed = kwargs.pop("seed", None)
    return make_detpmp_env(env_id=kwargs.pop("name"), wrappers=kwargs.pop("wrappers"), seed=seed,
                           mp_kwargs=kwargs.pop("mp_kwargs"), **kwargs)


def make_contextual_env(env_id, context, seed, rank):
    env = make_env(env_id, seed + rank, context=context)
    # env = gym.make(env_id, context=context)
    # env.seed(seed + rank)
    return lambda: env


def verify_time_limit(mp_time_limit: Union[None, float], env_time_limit: Union[None, float]):
    """
    When using DMC check if a manually specified time limit matches the trajectory duration the MP receives.
    Mostly, the time_limit for DMC is not specified and the default values from DMC are taken.
    This check, however, can only been done after instantiating the environment.
    It can be found in the BaseMP class.

    Args:
        mp_time_limit: max trajectory length of mp in seconds
        env_time_limit: max trajectory length of DMC environment in seconds

    Returns:

    """
    if mp_time_limit is not None and env_time_limit is not None:
        assert mp_time_limit == env_time_limit, \
            f"The manually specified 'time_limit' of {env_time_limit}s does not match " \
            f"the duration of {mp_time_limit}s for the MP."
