from typing import Tuple, Union

import copy
import importlib
import numpy as np
from fancy_gym.utils.make_env_helpers import make_bb
from fancy_gym.utils.utils import nested_update

from gymnasium import register as gym_register
from gymnasium import gym_make

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


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
        "wrappers": [],
        "trajectory_generator_kwargs": {
            'trajectory_generator_type': 'promp'
        },
        "phase_generator_kwargs": {
            'phase_generator_type': 'linear'
        },
        "controller_kwargs": {
            'controller_type': 'motor',
            "p_gains": 1.0,
            "d_gains": 0.1,
        },
        "basis_generator_kwargs": {
            'basis_generator_type': 'zero_rbf',
            'num_basis': 5,
            'num_basis_zero_start': 1,
            'basis_bandwidth_factor': 3.0,
        },
        "black_box_kwargs": {
        }
    },
    'DMP': {
        "wrappers": [],
        "trajectory_generator_kwargs": {
            'trajectory_generator_type': 'dmp'
        },
        "phase_generator_kwargs": {
            'phase_generator_type': 'exp'
        },
        "controller_kwargs": {
            'controller_type': 'motor',
            "p_gains": 1.0,
            "d_gains": 0.1,
        },
        "basis_generator_kwargs": {
            'basis_generator_type': 'rbf',
            'num_basis': 5
        },
        "black_box_kwargs": {
        }
    },
    'ProDMP': {
        "wrappers": [],
        "trajectory_generator_kwargs": {
            'trajectory_generator_type': 'prodmp',
            'duration': 2.0,
            'weights_scale': 1.0,
        },
        "phase_generator_kwargs": {
            'phase_generator_type': 'exp',
            'tau': 1.5,
        },
        "controller_kwargs": {
            'controller_type': 'motor',
            "p_gains": 1.0,
            "d_gains": 0.1,
        },
        "basis_generator_kwargs": {
            'basis_generator_type': 'prodmp',
            'alpha': 10,
            'num_basis': 5,
        },
        "black_box_kwargs": {
        }
    }
}

KNOWN_MPS = list(_BB_DEFAULTS.keys())
ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {mp_type: [] for mp_type in KNOWN_MPS}


def register(
        id,
        entry_point,
        mp_wrapper=DefaultMPWrapper,
        register_step_based=True,  # TODO: Detect
        add_mp_types=KNOWN_MPS,
        mp_config_override={},
        **kwargs
):
    if not callable(mp_wrapper):  # mp_wrapper can be given as a String (same notation as for entry_point)
        mod_name, attr_name = mp_wrapper.split(":")
        mod = importlib.import_module(mod_name)
        mp_wrapper = getattr(mod, attr_name)
    if register_step_based:
        gym_register(id=id, entry_point=entry_point, **kwargs)
    register_mps(id, mp_wrapper, add_mp_types, mp_config_override)


def register_mps(id, mp_wrapper, add_mp_types=KNOWN_MPS, mp_config_override={}):
    for mp_type in add_mp_types:
        register_mp(id, mp_wrapper, mp_type, mp_config_override.get(mp_type, {}))


def register_mp(id, mp_wrapper, mp_type, mp_config_override={}):
    assert mp_type in KNOWN_MPS, 'Unknown mp_type'
    assert id not in ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type], f'The environment {id} is already registered for {mp_type}.'
    parts = id.split('-')
    assert len(parts) >= 2 and parts[-1].startswith('v'), 'Malformed env id, must end in -v{int}.'
    fancy_id = '-'.join(parts[:-1]+[mp_type, parts[-1]])
    register(
        id=fancy_id,
        entry_point=bb_env_constructor,
        kwargs={
            'underlying_id': id,
            'mp_wrapper': mp_wrapper,
            'mp_type': mp_type,
            '_mp_config_override_register': mp_config_override
        }
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type].append(fancy_id)


def bb_env_constructor(underlying_id, mp_wrapper, mp_type, mp_config_override={}, _mp_config_override_register={}, **kwargs):
    raw_underlying_env = gym_make(underlying_id, **kwargs)
    underlying_env = mp_wrapper(raw_underlying_env)

    mp_config = underlying_env.get('mp_config', {})
    active_mp_config = copy.deepcopy(mp_config.get(mp_type, {}))
    global_inherit_defaults = mp_config.get('inherit_defaults', True)
    inherit_defaults = active_mp_config.pop('inherit_defaults', global_inherit_defaults)

    config = copy.deepcopy(_BB_DEFAULTS[mp_type]) if inherit_defaults else {}
    nested_update(config, active_mp_config)
    nested_update(config, _mp_config_override_register)
    nested_update(config, mp_config_override)

    wrappers = config.pop("wrappers")

    traj_gen_kwargs = config.pop("trajectory_generator_kwargs", {})
    black_box_kwargs = config.pop('black_box_kwargs', {})
    contr_kwargs = config.pop("controller_kwargs", {})
    phase_kwargs = config.pop("phase_generator_kwargs", {})
    basis_kwargs = config.pop("basis_generator_kwargs", {})

    return make_bb(underlying_env,
                   wrappers=wrappers,
                   black_box_kwargs=black_box_kwargs,
                   traj_gen_kwargs=traj_gen_kwargs,
                   controller_kwargs=contr_kwargs,
                   phase_kwargs=phase_kwargs,
                   basis_kwargs=basis_kwargs,
                   **config)
