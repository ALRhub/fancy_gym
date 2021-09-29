from gym import register
from gym.wrappers import FlattenObservation

from . import classic_control, mujoco, robotics

ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "DetPMP": []}

# Short Continuous Mountain Car
register(
    id="MountainCarContinuous-v1",
    entry_point="gym.envs.classic_control:Continuous_MountainCarEnv",
    max_episode_steps=100,
    reward_threshold=90.0,
)

# Open AI
# Classic Control
register(
    id='ContinuousMountainCarDetPMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "alr_envs:MountainCarContinuous-v1",
        "wrappers": [classic_control.continuous_mountain_car.MPWrapper],
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
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DetPMP"].append("ContinuousMountainCarDetPMP-v1")

register(
    id='ContinuousMountainCarDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.classic_control:MountainCarContinuous-v0",
        "wrappers": [classic_control.continuous_mountain_car.MPWrapper],
        "mp_kwargs": {
            "num_dof": 1,
            "num_basis": 4,
            "duration": 19.98,
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
        "wrappers": [mujoco.reacher_v2.MPWrapper],
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
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
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
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
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
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
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
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
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
