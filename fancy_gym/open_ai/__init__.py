from copy import deepcopy

from ..envs.registry import register, upgrade

from . import mujoco
from .deprecated_needs_gym_robotics import robotics

upgrade(
    id='Reacher-v2',
    mp_wrapper=mujoco.reacher_v2.MPWrapper,
    add_mp_types=['ProMP'],
)

"""
The Fetch environments are not supported by gym anymore. A new repository (gym_robotics) is supporting the environments.
However, the usage and so on needs to be checked

register(
    id='FetchSlideDenseProMP-v1',
    entry_point='fancy_gym.utils.make_env_helpers:make_promp_env_helper',
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
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchSlideDenseProMP-v1")

register(
    id='FetchSlideProMP-v1',
    entry_point='fancy_gym.utils.make_env_helpers:make_promp_env_helper',
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
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchSlideProMP-v1")

register(
    id='FetchReachDenseProMP-v1',
    entry_point='fancy_gym.utils.make_env_helpers:make_promp_env_helper',
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
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchReachDenseProMP-v1")

register(
    id='FetchReachProMP-v1',
    entry_point='fancy_gym.utils.make_env_helpers:make_promp_env_helper',
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
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("FetchReachProMP-v1")
"""
