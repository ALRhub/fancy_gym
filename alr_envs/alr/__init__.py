import numpy as np
from gym import register

from . import classic_control, mujoco
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from .mujoco.ball_in_a_cup.biac_pd import ALRBallInACupPDEnv
from .mujoco.reacher.alr_reacher import ALRReacherEnv
from .mujoco.reacher.balancing import BalancingEnv

from alr_envs.alr.mujoco.table_tennis.tt_gym import MAX_EPISODE_STEPS
from .mujoco.ant_jump.ant_jump import MAX_EPISODE_STEPS_ANTJUMP
from .mujoco.half_cheetah_jump.half_cheetah_jump import MAX_EPISODE_STEPS_HALFCHEETAHJUMP
from .mujoco.hopper_jump.hopper_jump import MAX_EPISODE_STEPS_HOPPERJUMP
from .mujoco.hopper_jump.hopper_jump_on_box import MAX_EPISODE_STEPS_HOPPERJUMPONBOX
from .mujoco.hopper_throw.hopper_throw import MAX_EPISODE_STEPS_HOPPERTHROW
from .mujoco.hopper_throw.hopper_throw_in_basket import MAX_EPISODE_STEPS_HOPPERTHROWINBASKET
from .mujoco.walker_2d_jump.walker_2d_jump import MAX_EPISODE_STEPS_WALKERJUMP

ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

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
    id='SimpleReacher-v1',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
        "random_start": False
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

register(
    id='LongSimpleReacher-v1',
    entry_point='alr_envs.alr.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False
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

# Mujoco

## Reacher
register(
    id='ALRReacher-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 5,
        "balance": False,
    }
)

register(
    id='ALRReacherSparse-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
        "balance": False,
    }
)

register(
    id='ALRReacherSparseOptCtrl-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherOptCtrlEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
        "balance": False,
    }
)

register(
    id='ALRReacherSparseBalanced-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
        "balance": True,
    }
)

register(
    id='ALRLongReacher-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 7,
        "balance": False,
    }
)

register(
    id='ALRLongReacherSparse-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 7,
        "balance": False,
    }
)

register(
    id='ALRLongReacherSparseBalanced-v0',
    entry_point='alr_envs.alr.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 7,
        "balance": True,
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
    id='ALRAntJump-v1',
    entry_point='alr_envs.alr.mujoco:ALRAntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_ANTJUMP,
        "context": True
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
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHalfCheetahJump-v1',
    entry_point='alr_envs.alr.mujoco:ALRHalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
        "context": True
    }
)
# CtxtFree are v0, Contextual are v1
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
register(
    id='ALRHopperJump-v1',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "context": True
    }
)

register(
    id='ALRHopperJump-v2',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpRndmPosEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP
    }
)

register(
    id='ALRHopperJump-v3',
    entry_point='alr_envs.alr.mujoco:ALRHopperXYJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "context": True,
        "healthy_reward": 1.0
    }
)

##### Hopper Jump step based reward
register(
    id='ALRHopperJump-v4',
    entry_point='alr_envs.alr.mujoco:ALRHopperXYJumpEnvStepBased',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMP,
        "context": True,
        "healthy_reward": 1.0
    }
)


# CtxtFree are v0, Contextual are v1
register(
    id='ALRHopperJumpOnBox-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHopperJumpOnBox-v1',
    entry_point='alr_envs.alr.mujoco:ALRHopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
        "context": True
    }
)
# CtxtFree are v0, Contextual are v1

register(
    id='ALRHopperThrow-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROW,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHopperThrow-v1',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROW,
        "context": True
    }
)
# CtxtFree are v0, Contextual are v1

register(
    id='ALRHopperThrowInBasket-v0',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRHopperThrowInBasket-v1',
    entry_point='alr_envs.alr.mujoco:ALRHopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
        "context": True
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRWalker2DJump-v0',
    entry_point='alr_envs.alr.mujoco:ALRWalker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_WALKERJUMP,
        "context": False
    }
)
# CtxtFree are v0, Contextual are v1
register(
    id='ALRWalker2DJump-v1',
    entry_point='alr_envs.alr.mujoco:ALRWalker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    kwargs={
        "max_episode_steps": MAX_EPISODE_STEPS_WALKERJUMP,
        "context": True
    }
)

## Balancing Reacher

register(
    id='Balancing-v0',
    entry_point='alr_envs.alr.mujoco:BalancingEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

## Table Tennis
register(id='TableTennis2DCtxt-v0',
         entry_point='alr_envs.alr.mujoco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 2})

register(id='TableTennis2DCtxt-v1',
         entry_point='alr_envs.alr.mujoco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 2, 'fixed_goal': True})

register(id='TableTennis4DCtxt-v0',
         entry_point='alr_envs.alr.mujocco:TTEnvGym',
         max_episode_steps=MAX_EPISODE_STEPS,
         kwargs={'ctxt_dim': 4})

## BeerPong
# fixed goal cup position
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


# random goal cup position
register(
        id='ALRBeerPong-v1',
        entry_point='alr_envs.alr.mujoco:ALRBeerBongEnv',
        max_episode_steps=300,
        kwargs={
            "rndm_goal": True,
            "cup_goal_pos": [-0.3, -1.2],
            "frame_skip": 2
        }
    )

# random goal cup position
register(
        id='ALRBeerPong-v2',
        entry_point='alr_envs.alr.mujoco:ALRBeerBongEnvStepBased',
        max_episode_steps=300,
        kwargs={
            "rndm_goal": True,
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
            "mp_kwargs": {
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
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:{_v}",
            "wrappers": [classic_control.simple_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 2 if "long" not in _v.lower() else 5,
                "num_basis": 5,
                "duration": 2,
                "policy_type": "motor",
                "weights_scale": 1,
                "zero_start": True,
                "policy_kwargs": {
                    "p_gains": .6,
                    "d_gains": .075
                }
            }
        }
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
        "mp_kwargs": {
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

register(
    id="ViaPointReacherProMP-v0",
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"alr_envs:ViaPointReacher-v0",
        "wrappers": [classic_control.viapoint_reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 5,
            "num_basis": 5,
            "duration": 2,
            "policy_type": "velocity",
            "weights_scale": 1,
            "zero_start": True
        }
    }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ViaPointReacherProMP-v0")

## Hole Reacher
_versions = ["v0", "v1", "v2"]
for _v in _versions:
    _env_id = f'HoleReacherDMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:HoleReacher-{_v}",
            "wrappers": [classic_control.hole_reacher.MPWrapper],
            "mp_kwargs": {
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

    _env_id = f'HoleReacherProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:HoleReacher-{_v}",
            "wrappers": [classic_control.hole_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5,
                "num_basis": 3,
                "duration": 2,
                "policy_type": "velocity",
                "weights_scale": 5,
                "zero_start": True
            }
        }
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
            "mp_kwargs": {
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
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:{_v}",
            "wrappers": [mujoco.reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5 if "long" not in _v.lower() else 7,
                "num_basis": 2,
                "duration": 4,
                "policy_type": "motor",
                "weights_scale": 5,
                "zero_start": True,
                "policy_kwargs": {
                    "p_gains": 1,
                    "d_gains": 0.1
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# ## Beerpong
# _versions = ["v0", "v1"]
# for _v in _versions:
#     _env_id = f'BeerpongProMP-{_v}'
#     register(
#         id=_env_id,
#         entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
#         kwargs={
#             "name": f"alr_envs:ALRBeerPong-{_v}",
#             "wrappers": [mujoco.beerpong.MPWrapper],
#             "mp_kwargs": {
#                 "num_dof": 7,
#                 "num_basis": 2,
#                 # "duration": 1,
#                 "duration": 0.5,
#                 # "post_traj_time": 2,
#                 "post_traj_time": 2.5,
#                 "policy_type": "motor",
#                 "weights_scale": 0.14,
#                 # "weights_scale": 1,
#                 "zero_start": True,
#                 "zero_goal": False,
#                 "policy_kwargs": {
#                     "p_gains": np.array([       1.5,   5,   2.55,    3,   2.,    2,   1.25]),
#                     "d_gains": np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125])
#                 }
#             }
#         }
#     )
#     ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## Beerpong
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'BeerpongProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRBeerPong-{_v}",
            "wrappers": [mujoco.beerpong.NewMPWrapper],
            "ep_wrapper_kwargs": {
                "weight_scale": 1
                },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 7
                },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 0.8,     # initial value
                'learn_tau': True,
                'learn_delay': False
                },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.array([1.5, 5, 2.55, 3, 2., 2, 1.25]),
                "d_gains": np.array([0.02333333, 0.1, 0.0625, 0.08, 0.03, 0.03, 0.0125]),
                },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 2,
                'num_basis_zero_start': 2
                }
            }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## Table Tennis
ctxt_dim = [2, 4]
for _v, cd in enumerate(ctxt_dim):
    _env_id = f'TableTennisProMP-v{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": "alr_envs:TableTennis{}DCtxt-v0".format(cd),
            "wrappers": [mujoco.table_tennis.MPWrapper],
            "mp_kwargs": {
                "num_dof": 7,
                "num_basis": 2,
                "duration": 1.25,
                "post_traj_time": 1.5,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": 0.5*np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0]),
                    "d_gains": 0.5*np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1])
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## AntJump
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRAntJumpProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRAntJump-{_v}",
            "wrappers": [mujoco.ant_jump.MPWrapper],
            "mp_kwargs": {
                "num_dof": 8,
                "num_basis": 5,
                "duration": 10,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(8),
                    "d_gains": 0.1*np.ones(8)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## AntJump
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRAntJumpProMP-{_v}'
    register(
        id= _env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRAntJump-{_v}",
            "wrappers": [mujoco.ant_jump.NewMPWrapper],
            "ep_wrapper_kwargs": {
                "weight_scale": 1
            },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 8
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 10,  # initial value
                'learn_tau': False,
                'learn_delay': False
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.ones(8),
                "d_gains": 0.1*np.ones(8),
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 2
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)



## HalfCheetahJump
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRHalfCheetahJumpProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRHalfCheetahJump-{_v}",
            "wrappers": [mujoco.half_cheetah_jump.MPWrapper],
            "mp_kwargs": {
                "num_dof": 6,
                "num_basis": 5,
                "duration": 5,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(6),
                    "d_gains": 0.1*np.ones(6)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# ## HopperJump
# _versions = ["v0", "v1"]
# for _v in _versions:
#     _env_id = f'ALRHopperJumpProMP-{_v}'
#     register(
#         id= _env_id,
#         entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
#         kwargs={
#             "name": f"alr_envs:ALRHopperJump-{_v}",
#             "wrappers": [mujoco.hopper_jump.MPWrapper],
#             "mp_kwargs": {
#                 "num_dof": 3,
#                 "num_basis": 5,
#                 "duration": 2,
#                 "post_traj_time": 0,
#                 "policy_type": "motor",
#                 "weights_scale": 1.0,
#                 "zero_start": True,
#                 "zero_goal": False,
#                 "policy_kwargs": {
#                     "p_gains": np.ones(3),
#                     "d_gains": 0.1*np.ones(3)
#                 }
#             }
#         }
#     )
#     ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# ## HopperJump
# register(
#     id= "ALRHopperJumpProMP-v2",
#     entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
#     kwargs={
#         "name": f"alr_envs:ALRHopperJump-v2",
#         "wrappers": [mujoco.hopper_jump.HighCtxtMPWrapper],
#         "mp_kwargs": {
#             "num_dof": 3,
#             "num_basis": 5,
#             "duration": 2,
#             "post_traj_time": 0,
#             "policy_type": "motor",
#             "weights_scale": 1.0,
#             "zero_start": True,
#             "zero_goal": False,
#             "policy_kwargs": {
#                 "p_gains": np.ones(3),
#                 "d_gains": 0.1*np.ones(3)
#             }
#         }
#     }
# )
# ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ALRHopperJumpProMP-v2")

## HopperJump
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRHopperJumpProMP-{_v}'
    register(
        id= _env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRHopperJump-{_v}",
            "wrappers": [mujoco.hopper_jump.NewMPWrapper],
            "ep_wrapper_kwargs": {
                "weight_scale": 1
            },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 3
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 2,  # initial value
                'learn_tau': False,
                'learn_delay': False
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.ones(3),
                "d_gains": 0.1*np.ones(3),
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 1
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## HopperJump
register(
    id= "ALRHopperJumpProMP-v2",
    entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
    kwargs={
        "name": f"alr_envs:ALRHopperJump-v2",
        "wrappers": [mujoco.hopper_jump.NewHighCtxtMPWrapper],
        "ep_wrapper_kwargs": {
                "weight_scale": 1
            },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 3
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 2,  # initial value
                'learn_tau': False,
                'learn_delay': False
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.ones(3),
                "d_gains": 0.1*np.ones(3),
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 1
            }
        }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ALRHopperJumpProMP-v2")


## HopperJump
register(
    id= "ALRHopperJumpProMP-v3",
    entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
    kwargs={
        "name": f"alr_envs:ALRHopperJump-v3",
        "wrappers": [mujoco.hopper_jump.NewMPWrapper],
        "ep_wrapper_kwargs": {
                "weight_scale": 1
            },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 3
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 2,  # initial value
                'learn_tau': False,
                'learn_delay': False
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.ones(3),
                "d_gains": 0.1*np.ones(3),
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 1
            }
        }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ALRHopperJumpProMP-v3")


## HopperJump
register(
    id= "ALRHopperJumpProMP-v4",
    entry_point='alr_envs.utils.make_env_helpers:make_mp_env_helper',
    kwargs={
        "name": f"alr_envs:ALRHopperJump-v4",
        "wrappers": [mujoco.hopper_jump.NewMPWrapper],
        "ep_wrapper_kwargs": {
                "weight_scale": 1
            },
            "movement_primitives_kwargs": {
                'movement_primitives_type': 'promp',
                'action_dim': 3
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'linear',
                'delay': 0,
                'tau': 2,  # initial value
                'learn_tau': False,
                'learn_delay': False
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": np.ones(3),
                "d_gains": 0.1*np.ones(3),
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 5,
                'num_basis_zero_start': 1
            }
        }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ALRHopperJumpProMP-v4")

## HopperJumpOnBox
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRHopperJumpOnBoxProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRHopperJumpOnBox-{_v}",
            "wrappers": [mujoco.hopper_jump.MPWrapper],
            "mp_kwargs": {
                "num_dof": 3,
                "num_basis": 5,
                "duration": 2,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(3),
                    "d_gains": 0.1*np.ones(3)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

#HopperThrow
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRHopperThrowProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRHopperThrow-{_v}",
            "wrappers": [mujoco.hopper_throw.MPWrapper],
            "mp_kwargs": {
                "num_dof": 3,
                "num_basis": 5,
                "duration": 2,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(3),
                    "d_gains": 0.1*np.ones(3)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## HopperThrowInBasket
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRHopperThrowInBasketProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRHopperThrowInBasket-{_v}",
            "wrappers": [mujoco.hopper_throw.MPWrapper],
            "mp_kwargs": {
                "num_dof": 3,
                "num_basis": 5,
                "duration": 2,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(3),
                    "d_gains": 0.1*np.ones(3)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

## Walker2DJump
_versions = ["v0", "v1"]
for _v in _versions:
    _env_id = f'ALRWalker2DJumpProMP-{_v}'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"alr_envs:ALRWalker2DJump-{_v}",
            "wrappers": [mujoco.walker_2d_jump.MPWrapper],
            "mp_kwargs": {
                "num_dof": 6,
                "num_basis": 5,
                "duration": 2.4,
                "post_traj_time": 0,
                "policy_type": "motor",
                "weights_scale": 1.0,
                "zero_start": True,
                "zero_goal": False,
                "policy_kwargs": {
                    "p_gains": np.ones(6),
                    "d_gains": 0.1*np.ones(6)
                }
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)