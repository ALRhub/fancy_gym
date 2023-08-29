from copy import deepcopy

from gym import register

from . import goal_object_change_mp_wrapper, goal_change_mp_wrapper, goal_endeffector_change_mp_wrapper, \
    object_change_mp_wrapper

ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": [], "ProDMP": []}

# MetaWorld

DEFAULT_BB_DICT_ProMP = {
    "name": 'EnvName',
    "wrappers": [],
    "trajectory_generator_kwargs": {
        'trajectory_generator_type': 'promp',
        'weights_scale': 10,
    },
    "phase_generator_kwargs": {
        'phase_generator_type': 'linear'
    },
    "controller_kwargs": {
        'controller_type': 'metaworld',
    },
    "basis_generator_kwargs": {
        'basis_generator_type': 'zero_rbf',
        'num_basis': 5,
        'num_basis_zero_start': 1
    },
    'black_box_kwargs': {
        'condition_on_desired': False,
    }
}

DEFAULT_BB_DICT_ProDMP = {
    "name": 'EnvName',
    "wrappers": [],
    "trajectory_generator_kwargs": {
        'trajectory_generator_type': 'prodmp',
        'auto_scale_basis': True,
        'weights_scale': 10,
        # 'goal_scale': 0.,
        'disable_goal': True,
    },
    "phase_generator_kwargs": {
        'phase_generator_type': 'exp',
        # 'alpha_phase' : 3,
    },
    "controller_kwargs": {
        'controller_type': 'metaworld',
    },
    "basis_generator_kwargs": {
        'basis_generator_type': 'prodmp',
        'num_basis': 5,
        'alpha': 10
    },
    'black_box_kwargs': {
        'condition_on_desired': False,
    }

}

_goal_change_envs = ["assembly-v2", "pick-out-of-hole-v2", "plate-slide-v2", "plate-slide-back-v2",
                     "plate-slide-side-v2", "plate-slide-back-side-v2"]
for _task in _goal_change_envs:
    task_id_split = _task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])

    # ProMP
    _env_id = f'{name}ProMP-{task_id_split[-1]}'
    kwargs_dict_goal_change_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_goal_change_promp['wrappers'].append(goal_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_change_promp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_change_promp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    # ProDMP
    _env_id = f'{name}ProDMP-{task_id_split[-1]}'
    kwargs_dict_goal_change_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_goal_change_prodmp['wrappers'].append(goal_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_change_prodmp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_change_prodmp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

_object_change_envs = ["bin-picking-v2", "hammer-v2", "sweep-into-v2"]
for _task in _object_change_envs:
    task_id_split = _task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])

    # ProMP
    _env_id = f'{name}ProMP-{task_id_split[-1]}'
    kwargs_dict_object_change_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_object_change_promp['wrappers'].append(object_change_mp_wrapper.MPWrapper)
    kwargs_dict_object_change_promp['name'] = f'metaworld:{_task}'
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_object_change_promp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    # ProDMP
    _env_id = f'{name}ProDMP-{task_id_split[-1]}'
    kwargs_dict_object_change_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_object_change_prodmp['wrappers'].append(object_change_mp_wrapper.MPWrapper)
    kwargs_dict_object_change_prodmp['name'] = f'metaworld:{_task}'
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_object_change_prodmp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

_goal_and_object_change_envs = ["box-close-v2", "button-press-v2", "button-press-wall-v2", "button-press-topdown-v2",
                                "button-press-topdown-wall-v2", "coffee-button-v2", "coffee-pull-v2",
                                "coffee-push-v2", "dial-turn-v2", "disassemble-v2", "door-close-v2",
                                "door-lock-v2", "door-open-v2", "door-unlock-v2", "hand-insert-v2",
                                "drawer-close-v2", "drawer-open-v2", "faucet-open-v2", "faucet-close-v2",
                                "handle-press-side-v2", "handle-press-v2", "handle-pull-side-v2",
                                "handle-pull-v2", "lever-pull-v2", "peg-insert-side-v2", "pick-place-wall-v2",
                                "reach-v2", "push-back-v2", "push-v2", "pick-place-v2", "peg-unplug-side-v2",
                                "soccer-v2", "stick-push-v2", "stick-pull-v2", "push-wall-v2", "reach-wall-v2",
                                "shelf-place-v2", "sweep-v2", "window-open-v2", "window-close-v2"
                                ]
for _task in _goal_and_object_change_envs:
    task_id_split = _task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])

    # ProMP
    _env_id = f'{name}ProMP-{task_id_split[-1]}'
    kwargs_dict_goal_and_object_change_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_goal_and_object_change_promp['wrappers'].append(goal_object_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_and_object_change_promp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_and_object_change_promp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    # ProDMP
    _env_id = f'{name}ProDMP-{task_id_split[-1]}'
    kwargs_dict_goal_and_object_change_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_goal_and_object_change_prodmp['wrappers'].append(goal_object_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_and_object_change_prodmp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_and_object_change_prodmp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

_goal_and_endeffector_change_envs = ["basketball-v2"]
for _task in _goal_and_endeffector_change_envs:
    task_id_split = _task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])

    # ProMP
    _env_id = f'{name}ProMP-{task_id_split[-1]}'
    kwargs_dict_goal_and_endeffector_change_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_goal_and_endeffector_change_promp['wrappers'].append(goal_endeffector_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_and_endeffector_change_promp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_and_endeffector_change_promp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    # ProDMP
    _env_id = f'{name}ProDMP-{task_id_split[-1]}'
    kwargs_dict_goal_and_endeffector_change_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_goal_and_endeffector_change_prodmp['wrappers'].append(goal_endeffector_change_mp_wrapper.MPWrapper)
    kwargs_dict_goal_and_endeffector_change_prodmp['name'] = f'metaworld:{_task}'

    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_goal_and_endeffector_change_prodmp
    )
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)
