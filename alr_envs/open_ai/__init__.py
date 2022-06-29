from gym import register
from gym.wrappers import FlattenObservation

from . import classic_control, mujoco, robotics

ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

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
    id='ContinuousMountainCarProMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "alr_envs:MountainCarContinuous-v1",
        "wrappers": [classic_control.continuous_mountain_car.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 1,
            "num_basis": 4,
            "duration": 2,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": 1.,
                "d_gains": 1.
            }
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ContinuousMountainCarProMP-v1")

register(
    id='ContinuousMountainCarProMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.classic_control:MountainCarContinuous-v0",
        "wrappers": [classic_control.continuous_mountain_car.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 1,
            "num_basis": 4,
            "duration": 19.98,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": 1.,
                "d_gains": 1.
            }
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ContinuousMountainCarProMP-v0")

register(
    id='ReacherProMP-v2',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.mujoco:Reacher-v2",
        "wrappers": [mujoco.reacher_v2.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 2,
            "num_basis": 6,
            "duration": 1,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": .6,
                "d_gains": .075
            }
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ReacherProMP-v2")

register(
    id='FetchSlideDenseProMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchSlideDense-v1",
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchSlideDenseProMP-v1")

register(
    id='FetchSlideProMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchSlide-v1",
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchSlideProMP-v1")

register(
    id='FetchReachDenseProMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchReachDense-v1",
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchReachDenseProMP-v1")

register(
    id='FetchReachProMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": "gym.envs.robotics:FetchReach-v1",
        "wrappers": [FlattenObservation, robotics.fetch.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 2,
            "post_traj_time": 0,
            "zero_start": True,
            "policy_type": "position"
        }
    }
)
ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchReachProMP-v1")
