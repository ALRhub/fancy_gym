from copy import deepcopy

import numpy as np
from gym import register

from alr_envs.alr.mujoco.table_tennis.tt_gym import MAX_EPISODE_STEPS
from . import classic_control, mujoco
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .mujoco.ant_jump.ant_jump import MAX_EPISODE_STEPS_ANTJUMP
from .mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from .mujoco.ball_in_a_cup.biac_pd import ALRBallInACupPDEnv
from .mujoco.half_cheetah_jump.half_cheetah_jump import MAX_EPISODE_STEPS_HALFCHEETAHJUMP
from .mujoco.hopper_jump.hopper_jump import MAX_EPISODE_STEPS_HOPPERJUMP
from .mujoco.hopper_jump.hopper_jump_on_box import MAX_EPISODE_STEPS_HOPPERJUMPONBOX
from .mujoco.hopper_throw.hopper_throw import MAX_EPISODE_STEPS_HOPPERTHROW
from .mujoco.hopper_throw.hopper_throw_in_basket import MAX_EPISODE_STEPS_HOPPERTHROWINBASKET
from .mujoco.reacher.reacher import ReacherEnv
from .mujoco.walker_2d_jump.walker_2d_jump import MAX_EPISODE_STEPS_WALKERJUMP

ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

DEFAULT_BB_DICT = {
    "name": 'EnvName',
    "wrappers": [],
    "trajectory_generator_kwargs": {
        'trajectory_generator_type': 'promp'
    },
    "phase_generator_kwargs": {
        'phase_generator_type': 'linear',
        'delay': 0,
        'learn_tau': False,
        'learn_delay': False
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

# Classic Control
## Simple Reacher
register(
    id='SimpleReacher-v0',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='LongSimpleReacher-v0',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

## Viapoint Reacher

register(
    id='ViaPointReacher-v0',
    entry_point='alr_envs.alr.classic_control:ViaPointReacherEnv',
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
    entry_point='alr_envs.alr.classic_control:HoleReacherEnv',
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
        entry_point='alr_envs.alr.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "n_links": _dims,
        }
    )

    register(
        id=f'Reacher{_dims}dSparse-v0',
        entry_point='alr_envs.alr.mujoco:ReacherEnv',
        max_episode_steps=200,
        kwargs={
            "sparse": True,
            "n_links": _dims,
        }
    )

## Hopper Jump random joints and desired position
register(
    id='HopperJumpSparse-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperXYJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        # "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "sparse": True,
        # "healthy_reward": 1.0
    }
)

## Hopper Jump random joints and desired position step based reward
register(
    id='HopperJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperXYJumpEnvStepBased',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        # "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "sparse": False,
        # "healthy_reward": 1.0
    }
)

register(
    id='ALRAntJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRAntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_ANTJUMP,
        "context": True
    }
)

register(
    id='ALRHalfCheetahJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRHalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
        "context": True
    }
)

register(
    id='HopperJumpOnBox-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
        "context": True
    }
)

register(
    id='ALRHopperThrow-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROW,
        "context": True
    }
)

register(
    id='ALRHopperThrowInBasket-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
        "context": True
    }
)

register(
    id='ALRWalker2DJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRWalker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_WALKERJUMP,
        "context": True
    }
)

## Table Tennis
register(id='TableTennis2DCtxt-v0',
         entry_point='alr_envs.alr.mujoco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 2})

register(id='TableTennis4DCtxt-v0',
         entry_point='alr_envs.alr.mujocco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 4})

register(
    id='ALRBeerPong-v0',
    entry_point='alr_envs.alr.mujoco:ALRBeerBongEnv',
    max_episode_steps=300,
    kwargs={
        "frame_skip": 2
    }
)

register(
    id='ALRBeerPong-v1',
    entry_point='alr_envs.alr.mujoco:ALRBeerBongEnv',
    max_episode_steps=300,
    kwargs={
        "frame_skip": 2
    }
)

# Here we use the same reward as in ALRBeerPong-v0, but now consider after the release,
# only one time step, i.e. we simulate until the end of th episode
register(
    id='ALRBeerPongStepBased-v0',
    entry_point='alr_envs.alr.mujoco:ALRBeerBongEnvStepBased',
    max_episode_steps=300,
    kwargs={
        "cup_goal_pos": [-0.3, -1.2],
        "frame_skip": 2
    }
)

# Beerpong with episodic reward, but fixed release time step
register(
    id='ALRBeerPongFixedRelease-v0',
    entry_point='alr_envs.alr.mujoco:ALRBeerBongEnvFixedReleaseStep',
    max_episode_steps=300,
    kwargs={
        "cup_goal_pos": [-0.3, -1.2],
        "frame_skip": 2
    }
)

# Motion Primitive Environments

## Simple Reacher
_versions = ["SimpleReacher-v0", "SimpleReacher-v1", "LongSimpleReacher-v0", "LongSimpleReacher-v1"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:{_v}",
            "wrappers": [classic_control.simple_reacher.MPWrapper],
            "traj_gen_kwargs": {
                "num_dof": 2 if "long" not in _v.lower() else 5,
                "num_basis": 5,
                "duration": 2,
                "alpha_phase": 2,
                "learn_goal": True,
                "policy_type": "motor",
                "weights_scale": 50,
                "policy_kwargs": {
                    "p_gains": .6,
                    "d_gains": .075
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_simple_reacher_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_simple_reacher_promp['wrappers'].append(classic_control.simple_reacher.MPWrapper)
    kwargs_dict_simple_reacher_promp['controller_kwargs']['p_gains'] = 0.6
    kwargs_dict_simple_reacher_promp['controller_kwargs']['d_gains'] = 0.075
    kwargs_dict_simple_reacher_promp['name'] = _env_id
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs=kwargs_dict_simple_reacher_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# Viapoint reacher
register(
    id='ViaPointReacherDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:ViaPointReacher-v0",
        "wrappers": [classic_control.viapoint_reacher.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 5,
            "num_basis": 5,
            "duration": 2,
            "learn_goal": True,
            "alpha_phase": 2,
            "policy_type": "velocity",
            "weights_scale": 50,
        }
    }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append("ViaPointReacherDMP-v0")

kwargs_dict_via_point_reacher_promp = deepcopy(DEFAULT_BB_DICT)
kwargs_dict_via_point_reacher_promp['wrappers'].append(classic_control.viapoint_reacher.MPWrapper)
kwargs_dict_via_point_reacher_promp['controller_kwargs']['controller_type'] = 'velocity'
kwargs_dict_via_point_reacher_promp['name'] = "ViaPointReacherProMP-v0"
register(
    id="ViaPointReacherProMP-v0",
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs=kwargs_dict_via_point_reacher_promp
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ViaPointReacherProMP-v0")

## Hole Reacher
_versions = ["HoleReacher-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:HoleReacher-{_v}",
            "wrappers": [classic_control.hole_reacher.MPWrapper],
            "traj_gen_kwargs": {
                "num_dof": 5,
                "num_basis": 5,
                "duration": 2,
                "learn_goal": True,
                "alpha_phase": 2.5,
                "bandwidth_factor": 2,
                "policy_type": "velocity",
                "weights_scale": 50,
                "goal_scale": 0.1
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_hole_reacher_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_hole_reacher_promp['wrappers'].append(classic_control.hole_reacher.MPWrapper)
    kwargs_dict_hole_reacher_promp['trajectory_generator_kwargs']['weight_scale'] = 2
    kwargs_dict_hole_reacher_promp['controller_kwargs']['controller_type'] = 'velocity'
    kwargs_dict_hole_reacher_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs=kwargs_dict_hole_reacher_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## ALRReacher
_versions = ["ALRReacher-v0", "ALRLongReacher-v0", "ALRReacherSparse-v0", "ALRLongReacherSparse-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}DMP-{_name[1]}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:{_v}",
            "wrappers": [mujoco.reacher.MPWrapper],
            "traj_gen_kwargs": {
                "num_dof": 5 if "long" not in _v.lower() else 7,
                "num_basis": 2,
                "duration": 4,
                "alpha_phase": 2,
                "learn_goal": True,
                "policy_type": "motor",
                "weights_scale": 5,
                "policy_kwargs": {
                    "p_gains": 1,
                    "d_gains": 0.1
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_alr_reacher_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_alr_reacher_promp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_alr_reacher_promp['controller_kwargs']['p_gains'] = 1
    kwargs_dict_alr_reacher_promp['controller_kwargs']['d_gains'] = 0.1
    kwargs_dict_alr_reacher_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs=kwargs_dict_alr_reacher_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

########################################################################################################################
## Beerpong ProMP
_versions = ['ALRBeerPong-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_bp_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_bp_promp['wrappers'].append(mujoco.beerpong.MPWrapper)
    kwargs_dict_bp_promp['phase_generator_kwargs']['learn_tau'] = True
    kwargs_dict_bp_promp['controller_kwargs']['p_gains'] = np.array([1.5, 5, 2.55, 3, 2., 2, 1.25])
    kwargs_dict_bp_promp['controller_kwargs']['d_gains'] = np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125])
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis'] = 2
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis_zero_start'] = 2
    kwargs_dict_bp_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs=kwargs_dict_bp_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

### BP with Fixed release
_versions = ["ALRBeerPongStepBased-v0", "ALRBeerPongFixedRelease-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_bp_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_bp_promp['wrappers'].append(mujoco.beerpong.MPWrapper)
    kwargs_dict_bp_promp['phase_generator_kwargs']['tau'] = 0.62
    kwargs_dict_bp_promp['controller_kwargs']['p_gains'] = np.array([1.5, 5, 2.55, 3, 2., 2, 1.25])
    kwargs_dict_bp_promp['controller_kwargs']['d_gains'] = np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125])
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis'] = 2
    kwargs_dict_bp_promp['basis_generator_kwargs']['num_basis_zero_start'] = 2
    kwargs_dict_bp_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs=kwargs_dict_bp_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)
########################################################################################################################

## Table Tennis needs to be fixed according to Zhou's implementation

########################################################################################################################

## AntJump
_versions = ['ALRAntJump-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_ant_jump_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_ant_jump_promp['wrappers'].append(mujoco.ant_jump.MPWrapper)
    kwargs_dict_ant_jump_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs=kwargs_dict_ant_jump_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

########################################################################################################################

## HalfCheetahJump
_versions = ['ALRHalfCheetahJump-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_halfcheetah_jump_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_halfcheetah_jump_promp['wrappers'].append(mujoco.half_cheetah_jump.MPWrapper)
    kwargs_dict_halfcheetah_jump_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs=kwargs_dict_halfcheetah_jump_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

########################################################################################################################


## HopperJump
_versions = ['ALRHopperJump-v0', 'ALRHopperJumpRndmJointsDesPos-v0', 'ALRHopperJumpRndmJointsDesPosStepBased-v0',
             'ALRHopperJumpOnBox-v0', 'ALRHopperThrow-v0', 'ALRHopperThrowInBasket-v0']
# TODO: Check if all environments work with the same MPWrapper
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_hopper_jump_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_hopper_jump_promp['wrappers'].append(mujoco.hopper_jump.MPWrapper)
    kwargs_dict_hopper_jump_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs=kwargs_dict_hopper_jump_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

########################################################################################################################


## Walker2DJump
_versions = ['ALRWalker2DJump-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_walker2d_jump_promp = deepcopy(DEFAULT_BB_DICT)
    kwargs_dict_walker2d_jump_promp['wrappers'].append(mujoco.walker_2d_jump.MPWrapper)
    kwargs_dict_walker2d_jump_promp['name'] = f"alr_envs:{_v}"
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs=kwargs_dict_walker2d_jump_promp
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

### Depricated, we will not provide non random starts anymore
"""
register(
    id='SimpleReacher-v1',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
        "random_start": False
    }
)

register(
    id='LongSimpleReacher-v1',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False
    }
)
register(
    id='HoleReacher-v1',
    entry_point='alr_envs.alr.classic_control:HoleReacherEnv',
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
    entry_point='alr_envs.alr.classic_control:HoleReacherEnv',
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
    entry_point='alr_envs.alr.mujoco:ALRAntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_ANTJUMP,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHalfCheetahJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRHalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
        "context": False
    }
)
register(
    id='ALRHopperJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpEnv',
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
        entry_point='alr_envs.alr.mujoco:ReacherEnv',
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
        entry_point='alr_envs.alr.mujoco:ReacherEnv',
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
            "name": f"alr_envs:{_env_id.replace('ProMP', '')}",
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
            "name": f"alr_envs:{_env_id.replace('ProMP', '')}",
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
        entry_point='alr_envs.alr.mujoco:ALRHopperJumpOnBoxEnv',
        max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
        kwargs={
            "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
            "context": False
        }
    )
    register(
    id='ALRHopperThrow-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROW,
        "context": False
    }
    )   
    register(
    id='ALRHopperThrowInBasket-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
        "context": False
    }
    )
    register(
    id='ALRWalker2DJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRWalker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_WALKERJUMP,
        "context": False
    }
    )
    register(id='TableTennis2DCtxt-v1',
         entry_point='alr_envs.alr.mujoco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 2, 'fixed_goal': True})

    register(
        id='ALRBeerPong-v0',
        entry_point='alr_envs.alr.mujoco:ALRBeerBongEnv',
        max_episode_steps=300,
        kwargs={
            "rndm_goal": False,
            "cup_goal_pos": [0.1, -2.0],
            "frame_skip": 2
        }
        )
"""
