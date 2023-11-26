from typing import Iterable, Type, Union, Optional

from copy import deepcopy

from ..envs.registry import register

from . import goal_object_change_mp_wrapper, goal_change_mp_wrapper, goal_endeffector_change_mp_wrapper, \
    object_change_mp_wrapper

try:
    import metaworld
except ModuleNotFoundError:
    print('[FANCY GYM] Metaworld not avaible.')
else:
    # Will only get executed, if import succeeds

    from . import metaworld_adapter

    metaworld_adapter.register_all_ML1()

    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": [], "ProDMP": []}

    # MetaWorld
    _goal_change_envs = ["assembly-v2", "pick-out-of-hole-v2", "plate-slide-v2", "plate-slide-back-v2",
                        "plate-slide-side-v2", "plate-slide-back-side-v2"]
    for _task in _goal_change_envs:
        register(
            id=f'metaworld/{_task}',
            register_step_based=False,
            mp_wrapper=goal_change_mp_wrapper.MPWrapper,
            add_mp_types=['ProMP', 'ProDMP'],
        )

    _object_change_envs = ["bin-picking-v2", "hammer-v2", "sweep-into-v2"]
    for _task in _object_change_envs:
        register(
            id=f'metaworld/{_task}',
            register_step_based=False,
            mp_wrapper=object_change_mp_wrapper.MPWrapper,
            add_mp_types=['ProMP', 'ProDMP'],
        )

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
        register(
            id=f'metaworld/{_task}',
            register_step_based=False,
            mp_wrapper=goal_object_change_mp_wrapper.MPWrapper,
            add_mp_types=['ProMP', 'ProDMP'],
        )

    _goal_and_endeffector_change_envs = ["basketball-v2"]
    for _task in _goal_and_endeffector_change_envs:
        register(
            id=f'metaworld/{_task}',
            register_step_based=False,
            mp_wrapper=goal_endeffector_change_mp_wrapper.MPWrapper,
            add_mp_types=['ProMP', 'ProDMP'],
        )