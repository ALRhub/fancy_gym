from typing import Tuple, Union, Callable, List, Dict, Any, Optional

import copy
import importlib
import numpy as np
from collections import defaultdict

from collections.abc import Mapping, MutableMapping

from fancy_gym.utils.make_env_helpers import make_bb
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper

from gymnasium import register as gym_register
from gymnasium import make as gym_make
from gymnasium.envs.registration import registry as gym_registry


class DefaultMPWrapper(RawInterfaceWrapper):
    @property
    def context_mask(self):
        """
            Returns boolean mask of the same shape as the observation space.
            It determines whether the observation is returned for the contextual case or not.
            This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
            E.g. Velocities starting at 0 are only changing after the first action. Given we only receive the
            context/part of the first observation, the velocities are not necessary in the observation for the task.
            Returns:
                bool array representing the indices of the observations
        """
        # If the env already defines a context_mask, we will use that
        if hasattr(self.env, 'context_mask'):
            return self.env.context_mask

        # Otherwise we will use the whole observation as the context. (Write a custom MPWrapper to change this behavior)
        return np.full(self.env.observation_space.shape, True)

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current position of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using velocity control,
            it should, however, be implemented regardless.
            E.g. The joint positions that are directly or indirectly controlled by the action.
        """
        assert hasattr(self.env, 'current_pos'), 'DefaultMPWrapper was unable to access env.current_pos. Please write a custom MPWrapper (recommended) or expose this attribute directly.'
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current velocity of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using position control,
            it should, however, be implemented regardless.
            E.g. The joint velocities that are directly or indirectly controlled by the action.
        """
        assert hasattr(self.env, 'current_vel'), 'DefaultMPWrapper was unable to access env.current_vel. Please write a custom MPWrapper (recommended) or expose this attribute directly.'
        return self.env.current_vel


_BB_DEFAULTS = {
    'ProMP': {
        'wrappers': [],
        'trajectory_generator_kwargs': {
            'trajectory_generator_type': 'promp'
        },
        'phase_generator_kwargs': {
            'phase_generator_type': 'linear'
        },
        'controller_kwargs': {
            'controller_type': 'motor',
            'p_gains': 1.0,
            'd_gains': 0.1,
        },
        'basis_generator_kwargs': {
            'basis_generator_type': 'zero_rbf',
            'num_basis': 5,
            'num_basis_zero_start': 1,
            'basis_bandwidth_factor': 3.0,
        },
        'black_box_kwargs': {
        }
    },
    'DMP': {
        'wrappers': [],
        'trajectory_generator_kwargs': {
            'trajectory_generator_type': 'dmp'
        },
        'phase_generator_kwargs': {
            'phase_generator_type': 'exp'
        },
        'controller_kwargs': {
            'controller_type': 'motor',
            'p_gains': 1.0,
            'd_gains': 0.1,
        },
        'basis_generator_kwargs': {
            'basis_generator_type': 'rbf',
            'num_basis': 5
        },
        'black_box_kwargs': {
        }
    },
    'ProDMP': {
        'wrappers': [],
        'trajectory_generator_kwargs': {
            'trajectory_generator_type': 'prodmp',
            'duration': 2.0,
            'weights_scale': 1.0,
        },
        'phase_generator_kwargs': {
            'phase_generator_type': 'exp',
            'tau': 1.5,
        },
        'controller_kwargs': {
            'controller_type': 'motor',
            'p_gains': 1.0,
            'd_gains': 0.1,
        },
        'basis_generator_kwargs': {
            'basis_generator_type': 'prodmp',
            'alpha': 10,
            'num_basis': 5,
        },
        'black_box_kwargs': {
        }
    }
}

KNOWN_MPS = list(_BB_DEFAULTS.keys())
_KNOWN_MPS_PLUS_ALL = KNOWN_MPS + ['all']
ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {mp_type: [] for mp_type in _KNOWN_MPS_PLUS_ALL}
MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS = {}


def register(
        id: str,
        entry_point: Optional[Union[Callable, str]] = None,
        mp_wrapper: RawInterfaceWrapper = DefaultMPWrapper,
        register_step_based: bool = True,  # TODO: Detect
        add_mp_types: List[str] = KNOWN_MPS,
        mp_config_override: Dict[str, Any] = {},
        **kwargs
):
    """
    Registers a Gymnasium environment, including Movement Primitives (MP) versions.
    If you only want to register MP versions for an already registered environment, use fancy_gym.upgrade instead.

    Args:
        id (str): The unique identifier for the environment.
        entry_point (Optional[Union[Callable, str]]): The entry point for creating the environment.
        mp_wrapper (RawInterfaceWrapper): The MP wrapper for the environment.
        register_step_based (bool): Whether to also register the raw srtep-based version of the environment (default True).
        add_mp_types (List[str]): List of additional MP types to register.
        mp_config_override (Dict[str, Any]): Dictionary for overriding MP configuration.
        **kwargs: Additional keyword arguments which are passed to the environment constructor.

    Notes:
        - When `register_step_based` is True, the raw environment will also be registered to gymnasium otherwise only mp-versions will be registered.
        - `entry_point` can be given as a string, allowing the same notation as gymnasium.
        - If `id` already exists in the Gymnasium registry and `register_step_based` is True,
          a warning message will be printed, suggesting to set `register_step_based=False` or use `fancy_gym.upgrade`.

    Example:
        To register a step-based environment with Movement Primitive versions (will use default mp_wrapper):
        >>> register("MyEnv-v0", MyEnvClass"my_module:MyEnvClass")

        The entry point can also be provided as a string:
        >>> register("MyEnv-v0", "my_module:MyEnvClass")

    """
    if register_step_based and id in gym_registry:
        print(f'[Info] Gymnasium env with id "{id}" already exists. You should supply register_step_based=False or use fancy_gym.upgrade if you only want to register mp versions of an existing env.')
    if register_step_based:
        assert entry_point != None, 'You need to provide an entry-point, when registering step-based.'
    if not callable(mp_wrapper):  # mp_wrapper can be given as a String (same notation as for entry_point)
        mod_name, attr_name = mp_wrapper.split(':')
        mod = importlib.import_module(mod_name)
        mp_wrapper = getattr(mod, attr_name)
    if register_step_based:
        gym_register(id=id, entry_point=entry_point, **kwargs)
    upgrade(id, mp_wrapper, add_mp_types, mp_config_override)


def upgrade(
        id: str,
        mp_wrapper: RawInterfaceWrapper = DefaultMPWrapper,
        add_mp_types: List[str] = KNOWN_MPS,
        base_id: Optional[str] = None,
        mp_config_override: Dict[str, Any] = {},
):
    """
    Upgrades an existing Gymnasium environment to include Movement Primitives (MP) versions.
    We expect the raw step-based env to be already registered with gymnasium. Otherwise please use fancy_gym.register instead.

    Args:
        id (str): The unique identifier for the environment.
        mp_wrapper (RawInterfaceWrapper): The MP wrapper for the environment (default is DefaultMPWrapper).
        add_mp_types (List[str]): List of additional MP types to register (default is KNOWN_MPS).
        base_id (Optional[str]): The unique identifier for the environment to upgrade. Will use id if non is provided. Can be defined to allow multiple registrations of different versions for the same step-based environment.
        mp_config_override (Dict[str, Any]): Dictionary for overriding MP configuration.

    Notes:
        - The `id` parameter should match the ID of the existing Gymnasium environment you wish to upgrade. You can also pick a new one, but then `base_id` needs to be provided.
        - The `mp_wrapper` parameter specifies the MP wrapper to use, allowing for customization.
        - `add_mp_types` can be used to specify additional MP types to register alongside the base environment.
        - The `base_id` parameter should match the ID of the existing Gymnasium environment you wish to upgrade.
        - `mp_config_override` allows for customizing MP configuration if needed.

    Example:
        To upgrade an existing environment with MP versions:
        >>> upgrade("MyEnv-v0", mp_wrapper=CustomMPWrapper)

        To upgrade an existing environment with custom MP types and configuration:
        >>> upgrade("MyEnv-v0", mp_wrapper=CustomMPWrapper, add_mp_types=["ProDMP", "DMP"], mp_config_override={"param": 42})
    """
    if not base_id:
        base_id = id
    register_mps(id, base_id, mp_wrapper, add_mp_types, mp_config_override)


def register_mps(id: str, base_id: str, mp_wrapper: RawInterfaceWrapper, add_mp_types: List[str] = KNOWN_MPS, mp_config_override: Dict[str, Any] = {}):
    for mp_type in add_mp_types:
        register_mp(id, base_id, mp_wrapper, mp_type, mp_config_override.get(mp_type, {}))


def register_mp(id: str, base_id: str, mp_wrapper: RawInterfaceWrapper, mp_type: List[str], mp_config_override: Dict[str, Any] = {}):
    assert mp_type in KNOWN_MPS, 'Unknown mp_type'
    assert id not in ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type], f'The environment {id} is already registered for {mp_type}.'

    parts = id.split('/')
    if len(parts) == 1:
        ns, name = 'gym', parts[0]
    elif len(parts) == 2:
        ns, name = parts[0], parts[1]
    else:
        raise ValueError('env id can not contain multiple "/".')

    parts = name.split('-')
    assert len(parts) >= 2 and parts[-1].startswith('v'), 'Malformed env id, must end in -v{int}.'

    fancy_id = f'{ns}_{mp_type}/{name}'

    gym_register(
        id=fancy_id,
        entry_point=bb_env_constructor,
        kwargs={
            'underlying_id': base_id,
            'mp_wrapper': mp_wrapper,
            'mp_type': mp_type,
            '_mp_config_override_register': mp_config_override
        }
    )

    ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type].append(fancy_id)
    ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS['all'].append(fancy_id)
    if ns not in MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS:
        MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS[ns] = {mp_type: [] for mp_type in _KNOWN_MPS_PLUS_ALL}
    MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS[ns][mp_type].append(fancy_id)
    MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS[ns]['all'].append(fancy_id)


def nested_update(base: MutableMapping, update):
    """
    Updated method for nested Mappings
    Args:
        base: main Mapping to be updated
        update: updated values for base Mapping

    """
    if any([item.endswith('_type') for item in update]):
        base = update
        return base
    for k, v in update.items():
        base[k] = nested_update(base.get(k, {}), v) if isinstance(v, Mapping) else v
    return base


def bb_env_constructor(underlying_id, mp_wrapper, mp_type, mp_config_override={}, _mp_config_override_register={}, **kwargs):
    raw_underlying_env = gym_make(underlying_id, **kwargs)
    underlying_env = mp_wrapper(raw_underlying_env)

    mp_config = getattr(underlying_env, 'mp_config') if hasattr(underlying_env, 'mp_config') else {}
    active_mp_config = copy.deepcopy(mp_config.get(mp_type, {}))
    global_inherit_defaults = mp_config.get('inherit_defaults', True)
    inherit_defaults = active_mp_config.pop('inherit_defaults', global_inherit_defaults)

    config = copy.deepcopy(_BB_DEFAULTS[mp_type]) if inherit_defaults else {}
    nested_update(config, active_mp_config)
    nested_update(config, _mp_config_override_register)
    nested_update(config, mp_config_override)

    wrappers = config.pop('wrappers')

    traj_gen_kwargs = config.pop('trajectory_generator_kwargs', {})
    black_box_kwargs = config.pop('black_box_kwargs', {})
    contr_kwargs = config.pop('controller_kwargs', {})
    phase_kwargs = config.pop('phase_generator_kwargs', {})
    basis_kwargs = config.pop('basis_generator_kwargs', {})

    return make_bb(underlying_env,
                   wrappers=wrappers,
                   black_box_kwargs=black_box_kwargs,
                   traj_gen_kwargs=traj_gen_kwargs,
                   controller_kwargs=contr_kwargs,
                   phase_kwargs=phase_kwargs,
                   basis_kwargs=basis_kwargs,
                   **config)
