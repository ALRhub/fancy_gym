from gym.envs.registration import register
from gym.wrappers import FlattenObservation

from alr_envs import classic_control, dmc, open_ai, meta

from alr_envs.utils.make_env_helpers import make_dmp_env
from alr_envs.utils.make_env_helpers import make_detpmp_env
from alr_envs.utils.make_env_helpers import make
from alr_envs.utils.make_env_helpers import make_rank

# Convenience function for all MP environments
ALL_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}
ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}

# Mujoco

## Reacher
register(
    id='ALRReacher-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 5,
        "balance": False,
    }
)

register(
    id='ALRReacherSparse-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
        "balance": False,
    }
)

register(
    id='ALRReacherSparseBalanced-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
        "balance": True,
    }
)

register(
    id='ALRLongReacher-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 7,
        "balance": False,
    }
)

register(
    id='ALRLongReacherSparse-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 7,
        "balance": False,
    }
)

register(
    id='ALRLongReacherSparseBalanced-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
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
    entry_point='alr_envs.mujoco:BalancingEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

# Classic control

## Simple Reacher
register(
    id='SimpleReacher-v0',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='SimpleReacher-v1',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
        "random_start": False
    }
)

register(
    id='LongSimpleReacher-v0',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

register(
    id='LongSimpleReacher-v1',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False
    }
)

## Viapoint Reacher

register(
    id='ViaPointReacher-v0',
    entry_point='alr_envs.classic_control:ViaPointReacher',
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
    entry_point='alr_envs.classic_control:HoleReacherEnv',
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
    entry_point='alr_envs.classic_control:HoleReacherEnv',
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
    entry_point='alr_envs.classic_control:HoleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": False,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": 0.25,
        "hole_depth": 1,
        "hole_x": 2,
        "collision_penalty": 100,
    }
)

# Motion Primitive Environments

## Simple Reacher
versions = ["SimpleReacher-v0", "SimpleReacher-v1", "LongSimpleReacher-v0", "LongSimpleReacher-v1"]
for v in versions:
    name = v.split("-")
    env_id = f'{name[0]}DMP-{name[1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:{v}",
            "wrappers": [classic_control.simple_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 2 if "long" not in v.lower() else 5,
                "num_basis": 5,
                "duration": 20,
                "alpha_phase": 2,
                "learn_goal": True,
                "policy_type": "velocity",
                "weights_scale": 50,
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(env_id)

    env_id = f'{name[0]}DetPMP-{name[1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:{v}",
            "wrappers": [classic_control.simple_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 2 if "long" not in v.lower() else 5,
                "num_basis": 5,
                "duration": 2,
                "width": 0.025,
                "policy_type": "velocity",
                "weights_scale": 0.2,
                "zero_start": True
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

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
    id='ViaPointReacherDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:ViaPointReacher-v0",
        "wrappers": [classic_control.viapoint_reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 5,
            "num_basis": 5,
            "duration": 2,
            "width": 0.025,
            "policy_type": "velocity",
            "weights_scale": 0.2,
            "zero_start": True
        }
    }
)
ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("ViaPointReacherDetPMP-v0")

## Hole Reacher
versions = ["v0", "v1", "v2"]
for v in versions:
    env_id = f'HoleReacherDMP-{v}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:HoleReacher-{v}",
            "wrappers": [classic_control.hole_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5,
                "num_basis": 5,
                "duration": 2,
                "learn_goal": True,
                "alpha_phase": 2,
                "bandwidth_factor": 2,
                "policy_type": "velocity",
                "weights_scale": 50,
                "goal_scale": 0.1
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(env_id)

    env_id = f'HoleReacherDetPMP-{v}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": f"alr_envs:HoleReacher-{v}",
            "wrappers": [classic_control.hole_reacher.MPWrapper],
            "mp_kwargs": {
                "num_dof": 5,
                "num_basis": 5,
                "duration": 2,
                "width": 0.025,
                "policy_type": "velocity",
                "weights_scale": 0.2,
                "zero_start": True
            }
        }
    )
    ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

## Deep Mind Control Suite (DMC)
### Suite

register(
    id=f'dmc_ball_in_cup-catch_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.ball_in_cup.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "learn_goal": True,
            "alpha_phase": 2,
            "bandwidth_factor": 2,
            "policy_type": "motor",
            "goal_scale": 0.1,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append("dmc_ball_in_cup-catch_dmp-v0")

register(
    id=f'dmc_ball_in_cup-catch_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.ball_in_cup.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "width": 0.025,
            "policy_type": "motor",
            "zero_start": True,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("dmc_ball_in_cup-catch_detpmp-v0")

register(
    id=f'dmc_reacher-easy_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "learn_goal": True,
            "alpha_phase": 2,
            "bandwidth_factor": 2,
            "policy_type": "motor",
            "weights_scale": 50,
            "goal_scale": 0.1,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append("dmc_reacher-easy_dmp-v0")

register(
    id=f'dmc_reacher-easy_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "width": 0.025,
            "policy_type": "motor",
            "weights_scale": 0.2,
            "zero_start": True,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("dmc_reacher-easy_detpmp-v0")

register(
    id=f'dmc_reacher-hard_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "learn_goal": True,
            "alpha_phase": 2,
            "bandwidth_factor": 2,
            "policy_type": "motor",
            "weights_scale": 50,
            "goal_scale": 0.1,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append("dmc_reacher-hard_dmp-v0")

register(
    id=f'dmc_reacher-hard_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "width": 0.025,
            "policy_type": "motor",
            "weights_scale": 0.2,
            "zero_start": True,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("dmc_reacher-hard_detpmp-v0")

dmc_cartpole_tasks = ["balance", "balance_sparse", "swingup", "swingup_sparse", "two_poles", "three_poles"]

for task in dmc_cartpole_tasks:
    env_id = f'dmc_cartpole-{task}_dmp-v0'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"cartpole-{task}",
            # "time_limit": 1,
            "camera_id": 0,
            "episode_length": 1000,
            "wrappers": [dmc.suite.cartpole.MPWrapper],
            "mp_kwargs": {
                "num_dof": 1,
                "num_basis": 5,
                "duration": 10,
                "learn_goal": True,
                "alpha_phase": 2,
                "bandwidth_factor": 2,
                "policy_type": "motor",
                "weights_scale": 50,
                "goal_scale": 0.1,
                "policy_kwargs": {
                    "p_gains": 10,
                    "d_gains": 10
                }
            }
        }
    )
    ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(env_id)

    env_id = f'dmc_cartpole-{task}_detpmp-v0'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": f"cartpole-{task}",
            # "time_limit": 1,
            "camera_id": 0,
            "episode_length": 1000,
            "wrappers": [dmc.suite.cartpole.MPWrapper],
            "mp_kwargs": {
                "num_dof": 1,
                "num_basis": 5,
                "duration": 10,
                "width": 0.025,
                "policy_type": "motor",
                "weights_scale": 0.2,
                "zero_start": True,
                "policy_kwargs": {
                    "p_gains": 10,
                    "d_gains": 10
                }
            }
        }
    )
    ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

### Manipulation

register(
    id=f'dmc_manipulation-reach_site_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [dmc.manipulation.reach.MPWrapper],
        "mp_kwargs": {
            "num_dof": 9,
            "num_basis": 5,
            "duration": 10,
            "learn_goal": True,
            "alpha_phase": 2,
            "bandwidth_factor": 2,
            "policy_type": "velocity",
            "weights_scale": 50,
            "goal_scale": 0.1,
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append("dmc_manipulation-reach_site_dmp-v0")

register(
    id=f'dmc_manipulation-reach_site_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [dmc.manipulation.reach.MPWrapper],
        "mp_kwargs": {
            "num_dof": 9,
            "num_basis": 5,
            "duration": 10,
            "width": 0.025,
            "policy_type": "velocity",
            "weights_scale": 0.2,
            "zero_start": True,
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("dmc_manipulation-reach_site_detpmp-v0")

## Open AI
register(
    id='ContinuousMountainCarDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.classic_control:MountainCarContinuous-v0",
        "wrappers": [open_ai.classic_control.continuous_mountain_car.MPWrapper],
        "mp_kwargs": {
            "num_dof": 1,
            "num_basis": 4,
            "duration": 2,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": 1.,
                "d_gains": 1.
            }
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("ContinuousMountainCarDetPMP-v0")

register(
    id='ReacherDetPMP-v2',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.mujoco:Reacher-v2",
        "wrappers": [open_ai.mujoco.reacher_v2.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 6,
            "duration": 1,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": .6,
                "d_gains": .075
            }
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("ReacherDetPMP-v2")

register(
    id='FetchSlideDenseDetPMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchSlideDense-v1",
        "wrappers": [FlattenObservation, open_ai.robotics.fetch.MPWrapper],
        "mp_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("FetchSlideDenseDetPMP-v1")

register(
    id='FetchSlideDetPMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchSlide-v1",
        "wrappers": [FlattenObservation, open_ai.robotics.fetch.MPWrapper],
        "mp_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("FetchSlideDetPMP-v1")

register(
    id='FetchReachDenseDetPMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchReachDense-v1",
        "wrappers": [FlattenObservation, open_ai.robotics.fetch.MPWrapper],
        "mp_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("FetchReachDenseDetPMP-v1")

register(
    id='FetchReachDetPMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchReach-v1",
        "wrappers": [FlattenObservation, open_ai.robotics.fetch.MPWrapper],
        "mp_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "width": 0.02,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("FetchReachDetPMP-v1")

# MetaWorld

goal_change_envs = ["assembly-v2", "pick-out-of-hole-v2", "plate-slide-v2", "plate-slide-back-v2",
                    ]
for task in goal_change_envs:
    task_id_split = task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])
    env_id = f'{name}DetPMP-{task_id_split[-1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": task,
            "wrappers": [meta.goal_change.MPWrapper],
            "mp_kwargs": {
                "num_dof": 4,
                "num_basis": 5,
                "duration": 6.25,
                "post_traj_time": 0,
                "width": 0.025,
                "zero_start": True,
                "policy_type": "metaworld",
            }
        }
    )
    ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

object_change_envs = ["bin-picking-v2", "hammer-v2", "sweep-into-v2"]
for task in object_change_envs:
    task_id_split = task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])
    env_id = f'{name}DetPMP-{task_id_split[-1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": env_id,
            "wrappers": [meta.object_change.MPWrapper],
            "mp_kwargs": {
                "num_dof": 4,
                "num_basis": 5,
                "duration": 6.25,
                "post_traj_time": 0,
                "width": 0.025,
                "zero_start": True,
                "policy_type": "metaworld",
            }
        }
    )
    ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

goal_and_object_change_envs = ["box-close-v2", "button-press-v2", "button-press-wall-v2", "button-press-topdown-v2",
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
for task in goal_and_object_change_envs:
    task_id_split = task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])
    env_id = f'{name}DetPMP-{task_id_split[-1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": env_id,
            "wrappers": [meta.goal_and_object_change.MPWrapper],
            "mp_kwargs": {
                "num_dof": 4,
                "num_basis": 5,
                "duration": 6.25,
                "post_traj_time": 0,
                "width": 0.025,
                "zero_start": True,
                "policy_type": "metaworld",
            }
        }
    )
    ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)

goal_and_endeffector_change_envs = ["basketball-v2"]
for task in goal_and_endeffector_change_envs:
    task_id_split = task.split("-")
    name = "".join([s.capitalize() for s in task_id_split[:-1]])
    env_id = f'{name}DetPMP-{task_id_split[-1]}'
    register(
        id=env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
        kwargs={
            "name": env_id,
            "wrappers": [meta.goal_and_endeffector_change.MPWrapper],
            "mp_kwargs": {
                "num_dof": 4,
                "num_basis": 5,
                "duration": 6.25,
                "post_traj_time": 0,
                "width": 0.025,
                "zero_start": True,
                "policy_type": "metaworld",
            }
        }
    )
    ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append(env_id)
