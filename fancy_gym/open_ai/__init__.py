from copy import deepcopy

from gym import register

from . import mujoco
from .deprecated_needs_gym_robotics import robotics

ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": [], "ProDMP": []}

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

kwargs_dict_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
kwargs_dict_reacher_promp['controller_kwargs']['p_gains'] = 0.6
kwargs_dict_reacher_promp['controller_kwargs']['d_gains'] = 0.075
kwargs_dict_reacher_promp['basis_generator_kwargs']['num_basis'] = 6
kwargs_dict_reacher_promp['name'] = "Reacher-v2"
kwargs_dict_reacher_promp['wrappers'].append(mujoco.reacher_v2.MPWrapper)
register(
    id='ReacherProMP-v2',
    entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
    kwargs=kwargs_dict_reacher_promp
)
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append("ReacherProMP-v2")
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
