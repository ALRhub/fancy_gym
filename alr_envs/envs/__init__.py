from copy import deepcopy

import numpy as np
from gym import register

from . import classic_control, mujoco
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .mujoco.ant_jump.ant_jump import MAX_EPISODE_STEPS_ANTJUMP
from .mujoco.half_cheetah_jump.half_cheetah_jump import MAX_EPISODE_STEPS_HALFCHEETAHJUMP
from .mujoco.hopper_jump.hopper_jump import MAX_EPISODE_STEPS_HOPPERJUMP
from .mujoco.hopper_jump.hopper_jump_on_box import MAX_EPISODE_STEPS_HOPPERJUMPONBOX
from .mujoco.hopper_throw.hopper_throw import MAX_EPISODE_STEPS_HOPPERTHROW
from .mujoco.hopper_throw.hopper_throw_in_basket import MAX_EPISODE_STEPS_HOPPERTHROWINBASKET
from .mujoco.beerpong.beerpong import MAX_EPISODE_STEPS_BEERPONG
from .mujoco.reacher.reacher import ReacherEnv
from .mujoco.walker_2d_jump.walker_2d_jump import MAX_EPISODE_STEPS_WALKERJUMP

ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

DEFAULT_BB_DICT_ProMP = {
    "name": 'EnvName',
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
        'num_basis_zero_start': 1
    }
}

DEFAULT_BB_DICT_DMP = {
    "name": 'EnvName',
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
    }
}

# Classic Control
## Simple Reacher
register(
    id='SimpleReacher-v0',
    entry_point='alr_envs.envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='LongSimpleReacher-v0',
    entry_point='alr_envs.envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

## Viapoint Reacher

register(
    id='ViaPointReacher-v0',
    entry_point='alr_envs.envs.classic_control:ViaPointReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "collision_penalty": 1000
    }
)

## Hole Reacher
register(
    id='HoleReacher-v0',
    entry_point='alr_envs.envs.classic_control:HoleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": True,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": None,
        "hole_depth": 1,
        "hole_x": None,
        "collision_penalty": 100,
    }
)

# Mujoco

## Reacher
for _dims in [5, 7]:
    register(
        id=f'Reacher{_dims}d-v0',
        entry_point='alr_envs.envs.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "n_links": _dims,
        }
    )

    register(
        id=f'Reacher{_dims}dSparse-v0',
        entry_point='alr_envs.envs.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "sparse": True,
            "n_links": _dims,
        }
    )

register(
    id='HopperJumpSparse-v0',
    entry_point='alr_envs.envs.mujoco:HopperJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": True,
    }
)

register(
    id='HopperJump-v0',
    entry_point='alr_envs.envs.mujoco:HopperJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": False,
        "healthy_reward": 1.0,
        "contact_weight": 0.0,
        "height_weight": 3.0,
    }
)

register(
    id='AntJump-v0',
    entry_point='alr_envs.envs.mujoco:AntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
)

register(
    id='HalfCheetahJump-v0',
    entry_point='alr_envs.envs.mujoco:HalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
)

register(
    id='HopperJumpOnBox-v0',
    entry_point='alr_envs.envs.mujoco:HopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
)

register(
    id='ALRHopperThrow-v0',
    entry_point='alr_envs.envs.mujoco:HopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
)

register(
    id='ALRHopperThrowInBasket-v0',
    entry_point='alr_envs.envs.mujoco:HopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
)

register(
    id='ALRWalker2DJump-v0',
    entry_point='alr_envs.envs.mujoco:Walker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
)

register(
    id='BeerPong-v0',
    entry_point='alr_envs.envs.mujoco:BeerPongEnv',
    max_episode_steps=MAX_EPISODE_STEPS_BEERPONG,
)

# Here we use the same reward as in BeerPong-v0, but now consider after the release,
# only one time step, i.e. we simulate until the end of th episode
register(
    id='BeerPongStepBased-v0',
    entry_point='alr_envs.envs.mujoco:BeerPongEnvStepBasedEpisodicReward',
    max_episode_steps=MAX_EPISODE_STEPS_BEERPONG,
)

# Beerpong with episodic reward, but fixed release time step
register(
    id='BeerPongFixedRelease-v0',
    entry_point='alr_envs.envs.mujoco:BeerPongEnvFixedReleaseStep',
    max_episode_steps=MAX_EPISODE_STEPS_BEERPONG,
)

# Motion Primitive Environments

## Simple Reacher
_versions = ["SimpleReacher-v0", "LongSimpleReacher-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    kwargs_dict_simple_reacher_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_simple_reacher_dmp['wrappers'].append(classic_control.simple_reacher.MPWrapper)
    kwargs_dict_simple_reacher_dmp['controller_kwargs']['p_gains'] = 0.6
    kwargs_dict_simple_reacher_dmp['controller_kwargs']['d_gains'] = 0.075
    kwargs_dict_simple_reacher_dmp['trajectory_generator_kwargs']['weight_scale'] = 50
    kwargs_dict_simple_reacher_dmp['phase_generator_kwargs']['alpha_phase'] = 2
    kwargs_dict_simple_reacher_dmp['name'] = f"{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_simple_reacher_dmp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_simple_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_simple_reacher_promp['wrappers'].append(classic_control.simple_reacher.MPWrapper)
    kwargs_dict_simple_reacher_promp['controller_kwargs']['p_gains'] = 0.6
    kwargs_dict_simple_reacher_promp['controller_kwargs']['d_gains'] = 0.075
    kwargs_dict_simple_reacher_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_simple_reacher_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# Viapoint reacher
kwargs_dict_via_point_reacher_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_via_point_reacher_dmp['wrappers'].append(classic_control.viapoint_reacher.MPWrapper)
kwargs_dict_via_point_reacher_dmp['controller_kwargs']['controller_type'] = 'velocity'
kwargs_dict_via_point_reacher_dmp['trajectory_generator_kwargs']['weight_scale'] = 50
kwargs_dict_via_point_reacher_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_via_point_reacher_dmp['name'] = "ViaPointReacher-v0"
register(
    id='ViaPointReacherDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
    # max_episode_steps=1,
    kwargs=kwargs_dict_via_point_reacher_dmp
)
ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["DMP"].append("ViaPointReacherDMP-v0")

kwargs_dict_via_point_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
kwargs_dict_via_point_reacher_promp['wrappers'].append(classic_control.viapoint_reacher.MPWrapper)
kwargs_dict_via_point_reacher_promp['controller_kwargs']['controller_type'] = 'velocity'
kwargs_dict_via_point_reacher_promp['name'] = "ViaPointReacher-v0"
register(
    id="ViaPointReacherProMP-v0",
    entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
    kwargs=kwargs_dict_via_point_reacher_promp
)
ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ViaPointReacherProMP-v0")

## Hole Reacher
_versions = ["HoleReacher-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    kwargs_dict_hole_reacher_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_hole_reacher_dmp['wrappers'].append(classic_control.hole_reacher.MPWrapper)
    kwargs_dict_hole_reacher_dmp['controller_kwargs']['controller_type'] = 'velocity'
    # TODO: Before it was weight scale 50 and goal scale 0.1. We now only have weight scale and thus set it to 500. Check
    kwargs_dict_hole_reacher_dmp['trajectory_generator_kwargs']['weight_scale'] = 500
    kwargs_dict_hole_reacher_dmp['phase_generator_kwargs']['alpha_phase'] = 2.5
    kwargs_dict_hole_reacher_dmp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        # max_episode_steps=1,
        kwargs=kwargs_dict_hole_reacher_dmp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_hole_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_hole_reacher_promp['wrappers'].append(classic_control.hole_reacher.MPWrapper)
    kwargs_dict_hole_reacher_promp['trajectory_generator_kwargs']['weight_scale'] = 2
    kwargs_dict_hole_reacher_promp['controller_kwargs']['controller_type'] = 'velocity'
    kwargs_dict_hole_reacher_promp['name'] = f"{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_hole_reacher_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## ReacherNd
_versions = ["Reacher5d-v0", "Reacher7d-v0", "Reacher5dSparse-v0", "Reacher7dSparse-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    kwargs_dict_reacherNd_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_reacherNd_dmp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_reacherNd_dmp['trajectory_generator_kwargs']['weight_scale'] = 5
    kwargs_dict_reacherNd_dmp['phase_generator_kwargs']['alpha_phase'] = 2
    kwargs_dict_reacherNd_dmp['basis_generator_kwargs']['num_basis'] = 2
    kwargs_dict_reacherNd_dmp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        # max_episode_steps=1,
        kwargs=kwargs_dict_reacherNd_dmp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_alr_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_alr_reacher_promp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_alr_reacher_promp['controller_kwargs']['p_gains'] = 1
    kwargs_dict_alr_reacher_promp['controller_kwargs']['d_gains'] = 0.1
    kwargs_dict_alr_reacher_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_alr_reacher_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

########################################################################################################################
## Beerpong ProMP
_versions = ['BeerPong-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_bp_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_bp_promp['wrappers'].append(mujoco.beerpong.MPWrapper)
    kwargs_dict_bp_promp['phase_generator_kwargs']['learn_tau'] = True
    kwargs_dict_bp_promp['controller_kwargs']['p_gains'] = np.array([1.5, 5, 2.55, 3, 2., 2, 1.25])
    kwargs_dict_bp_promp['controller_kwargs']['d_gains'] = np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125])
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis'] = 2
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis_zero_start'] = 2
    kwargs_dict_bp_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_bp_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

### BP with Fixed release
_versions = ["BeerPongStepBased-v0", "BeerPongFixedRelease-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_bp_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_bp_promp['wrappers'].append(mujoco.beerpong.MPWrapper)
    kwargs_dict_bp_promp['phase_generator_kwargs']['tau'] = 0.62
    kwargs_dict_bp_promp['controller_kwargs']['p_gains'] = np.array([1.5, 5, 2.55, 3, 2., 2, 1.25])
    kwargs_dict_bp_promp['controller_kwargs']['d_gains'] = np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125])
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis'] = 2
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis_zero_start'] = 2
    kwargs_dict_bp_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_bp_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)
########################################################################################################################

## Table Tennis needs to be fixed according to Zhou's implementation

# TODO: Add later when finished
# ########################################################################################################################
#
# ## AntJump
# _versions = ['ALRAntJump-v0']
# for _v in _versions:
#     _name = _v.split("-")
#     _env_id = f'{_name[0]}ProMP-{_name[1]}'
#     kwargs_dict_ant_jump_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
#     kwargs_dict_ant_jump_promp['wrappers'].append(mujoco.ant_jump.MPWrapper)
#     kwargs_dict_ant_jump_promp['name'] = _v
#     register(
#         id=_env_id,
#         entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
#         kwargs=kwargs_dict_ant_jump_promp
#     )
#     ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)
#
# ########################################################################################################################
#
# ## HalfCheetahJump
# _versions = ['ALRHalfCheetahJump-v0']
# for _v in _versions:
#     _name = _v.split("-")
#     _env_id = f'{_name[0]}ProMP-{_name[1]}'
#     kwargs_dict_halfcheetah_jump_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
#     kwargs_dict_halfcheetah_jump_promp['wrappers'].append(mujoco.half_cheetah_jump.MPWrapper)
#     kwargs_dict_halfcheetah_jump_promp['name'] = _v
#     register(
#         id=_env_id,
#         entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
#         kwargs=kwargs_dict_halfcheetah_jump_promp
#     )
#     ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)
#
# ########################################################################################################################


## HopperJump
_versions = ['HopperJump-v0', 'HopperJumpSparse-v0',
             # 'ALRHopperJumpOnBox-v0', 'ALRHopperThrow-v0', 'ALRHopperThrowInBasket-v0'
             ]
# TODO: Check if all environments work with the same MPWrapper
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_hopper_jump_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_hopper_jump_promp['wrappers'].append(mujoco.hopper_jump.MPWrapper)
    kwargs_dict_hopper_jump_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_hopper_jump_promp
    )
    ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# ########################################################################################################################
#
#
# ## Walker2DJump
# _versions = ['ALRWalker2DJump-v0']
# for _v in _versions:
#     _name = _v.split("-")
#     _env_id = f'{_name[0]}ProMP-{_name[1]}'
#     kwargs_dict_walker2d_jump_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
#     kwargs_dict_walker2d_jump_promp['wrappers'].append(mujoco.walker_2d_jump.MPWrapper)
#     kwargs_dict_walker2d_jump_promp['name'] = _v
#     register(
#         id=_env_id,
#         entry_point='alr_envs.utils.make_env_helpers:make_bb_env_helper',
#         kwargs=kwargs_dict_walker2d_jump_promp
#     )
#     ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

### Depricated, we will not provide non random starts anymore
"""
register(
    id='SimpleReacher-v1',
    entry_point='alr_envs.envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
        "random_start": False
    }
)

register(
    id='LongSimpleReacher-v1',
    entry_point='alr_envs.envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False
    }
)
register(
    id='HoleReacher-v1',
    entry_point='alr_envs.envs.classic_control:HoleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": 0.25,
        "hole_depth": 1,
        "hole_x": None,
        "collision_penalty": 100,
    }
)
register(
    id='HoleReacher-v2',
    entry_point='alr_envs.envs.classic_control:HoleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": 0.25,
        "hole_depth": 1,
        "hole_x": 2,
        "collision_penalty": 1,
    }
)

# CtxtFree are v0, Contextual are v1
register(
    id='ALRAntJump-v0',
    entry_point='alr_envs.envs.mujoco:AntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_ANTJUMP,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHalfCheetahJump-v0',
    entry_point='alr_envs.envs.mujoco:HalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
        "context": False
    }
)
register(
    id='ALRHopperJump-v0',
    entry_point='alr_envs.envs.mujoco:HopperJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "context": False,
        "healthy_reward": 1.0
    }
)

"""

### Deprecated used for CorL paper
"""
_vs = np.arange(101).tolist() + [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
for i in _vs:
    _env_id = f'ALRReacher{i}-v0'
    register(
        id=_env_id,
        entry_point='alr_envs.envs.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "steps_before_reward": 0,
            "n_links": 5,
            "balance": False,
            'ctrl_cost_weight': i
        }
    )

    _env_id = f'ALRReacherSparse{i}-v0'
    register(
        id=_env_id,
        entry_point='alr_envs.envs.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "steps_before_reward": 200,
            "n_links": 5,
            "balance": False,
            'ctrl_cost_weight': i
        }
    )
    _vs = np.arange(101).tolist() + [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
for i in _vs:
    _env_id = f'ALRReacher{i}ProMP-v0'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"{_env_id.replace('ProMP', '')}",
            "wrappers": [mujoco.reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5,
                "num_basis": 5,
                "duration": 4,
                "policy_type": "motor",
                # "weights_scale": 5,
                "n_zero_basis": 1,
                "zero_start": True,
                "policy_kwargs": {
                    "p_gains": 1,
                    "d_gains": 0.1
                }
            }
        }
    )

    _env_id = f'ALRReacherSparse{i}ProMP-v0'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"{_env_id.replace('ProMP', '')}",
            "wrappers": [mujoco.reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5,
                "num_basis": 5,
                "duration": 4,
                "policy_type": "motor",
                # "weights_scale": 5,
                "n_zero_basis": 1,
                "zero_start": True,
                "policy_kwargs": {
                    "p_gains": 1,
                    "d_gains": 0.1
                }
            }
        }
    )
    
    register(
        id='ALRHopperJumpOnBox-v0',
        entry_point='alr_envs.envs.mujoco:HopperJumpOnBoxEnv',
        max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
        kwargs={
            "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
            "context": False
        }
    )
    register(
    id='ALRHopperThrow-v0',
    entry_point='alr_envs.envs.mujoco:HopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROW,
        "context": False
    }
    )   
    register(
    id='ALRHopperThrowInBasket-v0',
    entry_point='alr_envs.envs.mujoco:HopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
        "context": False
    }
    )
    register(
    id='ALRWalker2DJump-v0',
    entry_point='alr_envs.envs.mujoco:Walker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_WALKERJUMP,
        "context": False
    }
    )
    register(id='TableTennis2DCtxt-v1',
         entry_point='alr_envs.envs.mujoco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 2, 'fixed_goal': True})

    register(
        id='ALRBeerPong-v0',
        entry_point='alr_envs.envs.mujoco:ALRBeerBongEnv',
        max_episode_steps=300,
        kwargs={
            "rndm_goal": False,
            "cup_goal_pos": [0.1, -2.0],
            "frame_skip": 2
        }
        )
"""