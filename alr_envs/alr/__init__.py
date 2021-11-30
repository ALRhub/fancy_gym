from gym import register

from . import classic_control, mujoco
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .mujoco.ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from .mujoco.ball_in_a_cup.biac_pd import ALRBallInACupPDEnv
from .mujoco.reacher.alr_reacher import ALRReacherEnv
from .mujoco.reacher.balancing import BalancingEnv

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

## Balancing Reacher

register(
    id='Balancing-v0',
    entry_point='alr_envs.alr.mujoco:BalancingEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
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
                "num_basis": 5,
                "duration": 2,
                "policy_type": "velocity",
                "weights_scale": 0.1,
                "zero_start": True
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)
