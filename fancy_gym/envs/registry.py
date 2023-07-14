from fancy_gym.utils.make_env_helpers import make_bb
from fancy_gym.utils.utils import nested_update

from gymnasium import register as gym_register
from gymnasium import gym_make

import copy

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
        register_step_based=True,  # TODO: Detect
        add_mp_types=KNOWN_MPS,
        override_mp_config={},
        **kwargs
):
    if register_step_based:
        gym_register(id=id, entry_point=entry_point, **kwargs)
    register_mps(id, override_mp_config, add_mp_types)


def register_mps(id, add_mp_types=KNOWN_MPS):
    for mp_type in add_mp_types:
        register_mp(id, mp_type)


def register_mp(id, mp_type):
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
            'mp_type': mp_type
        }
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS[mp_type].append(fancy_id)


def bb_env_constructor(underlying_id, mp_type, step_based_kwargs={}, mp_config_override={}):
    underlying_env = gym_make(underlying_id, **step_based_kwargs)
    env_metadata = underlying_env.metadata

    config = copy.deepcopy(_BB_DEFAULTS[mp_type])
    metadata_config = env_metadata.get('mp_config', {})
    nested_update(config, metadata_config)
    nested_update(config, mp_config_override)

    wrappers = config.pop("wrappers")

    traj_gen_kwargs = config.pop("trajectory_generator_kwargs", {})
    black_box_kwargs = config.pop('black_box_kwargs', {})
    contr_kwargs = config.pop("controller_kwargs", {})
    phase_kwargs = config.pop("phase_generator_kwargs", {})
    basis_kwargs = config.pop("basis_generator_kwargs", {})

    return make_bb(underlying_env, wrappers=wrappers,
                   black_box_kwargs=black_box_kwargs,
                   traj_gen_kwargs=traj_gen_kwargs, controller_kwargs=contr_kwargs,
                   phase_kwargs=phase_kwargs,
                   basis_kwargs=basis_kwargs, **config)
