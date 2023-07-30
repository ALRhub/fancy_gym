from typing import Tuple, Union

import copy
import importlib
import numpy as np
from collections import defaultdict

from fancy_gym.utils.make_env_helpers import make_bb
from fancy_gym.utils.utils import nested_update
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper

from gymnasium import register as gym_register
from gymnasium import make as gym_make
from gymnasium.envs.registration import registry as gym_registry


class DefaultMPWrapper(RawInterfaceWrapper):
    @property
    def context_mask(self):
        # If the env already defines a context_mask, we will use that
        if hasattr(self.env, 'context_mask'):
            return self.env.context_mask

        # Otherwise we will use the whole observation as the context. (Write a custom MPWrapper to change this behavior)
        return np.full(self.env.observation_space.shape, True)

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        assert hasattr(self.env, 'current_pos'), 'DefaultMPWrapper was unable to access env.current_pos. Please write a custom MPWrapper (recommended) or expose this attribute directly.'
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
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
        id,
        entry_point=None,
        mp_wrapper=DefaultMPWrapper,
        register_step_based=True,  # TODO: Detect
        add_mp_types=KNOWN_MPS,
        mp_config_override={},
        **kwargs
):
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
    register_mps(id, mp_wrapper, add_mp_types, mp_config_override)


def upgrade(
        id,
        mp_wrapper=DefaultMPWrapper,
        add_mp_types=KNOWN_MPS,
        mp_config_override={},
        **kwargs
):
    register(
        id,
        entry_point=None,
        mp_wrapper=mp_wrapper,
        register_step_based=False,
        add_mp_types=add_mp_types,
        mp_config_override=mp_config_override,
        **kwargs
    )


def register_mps(id, mp_wrapper, add_mp_types=KNOWN_MPS, mp_config_override={}):
    for mp_type in add_mp_types:
        register_mp(id, mp_wrapper, mp_type, mp_config_override.get(mp_type, {}))


def register_mp(id, mp_wrapper, mp_type, mp_config_override={}):
    assert mp_type in KNOWN_MPS, 'Unknown mp_type'
    assert id not in ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type], f'The environment {id} is already registered for {mp_type}.'

    parts = id.split('/')
    if len(parts) == 1:
        ns, name = 'gym', parts[0]
    elif len(parts) == 2:
        ns, name = parts[0], parts[1]
    else:
        raise ValueError('env id can not contain multiple "/".')

    parts = id.split('-')
    assert len(parts) >= 2 and parts[-1].startswith('v'), 'Malformed env id, must end in -v{int}.'
    fancy_id = '-'.join(parts[:-1]+[mp_type, parts[-1]])

    gym_register(
        id=fancy_id,
        entry_point=bb_env_constructor,
        kwargs={
            'underlying_id': id,
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
