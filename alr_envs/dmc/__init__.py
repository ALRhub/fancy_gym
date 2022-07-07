from copy import deepcopy

from . import manipulation, suite

ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

from gym.envs.registration import register

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


# DeepMind Control Suite (DMC)
kwargs_dict_bic_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_bic_dmp['name'] = f"ball_in_cup-catch"
kwargs_dict_bic_dmp['wrappers'].append(suite.ball_in_cup.MPWrapper)
kwargs_dict_bic_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_bic_dmp['trajectory_generator_kwargs']['weight_scale'] = 10  # TODO: weight scale 1, but goal scale 0.1
kwargs_dict_bic_dmp['controller_kwargs']['p_gains'] = 50
kwargs_dict_bic_dmp['controller_kwargs']['d_gains'] = 1
register(
    id=f'dmc_ball_in_cup-catch_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.ball_in_cup.MPWrapper],
        "traj_gen_kwargs": {
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

kwargs_dict_bic_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_bic_promp['name'] = f"ball_in_cup-catch"
kwargs_dict_bic_promp['wrappers'].append(suite.ball_in_cup.MPWrapper)
kwargs_dict_bic_promp['controller_kwargs']['p_gains'] = 50
kwargs_dict_bic_promp['controller_kwargs']['d_gains'] = 1
register(
    id=f'dmc_ball_in_cup-catch_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.ball_in_cup.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
            "policy_type": "motor",
            "zero_start": True,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("dmc_ball_in_cup-catch_promp-v0")

kwargs_dict_reacher_easy_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_reacher_easy_dmp['name'] = f"reacher-easy"
kwargs_dict_reacher_easy_dmp['wrappers'].append(suite.reacher.MPWrapper)
kwargs_dict_reacher_easy_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_reacher_easy_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
kwargs_dict_reacher_easy_dmp['controller_kwargs']['p_gains'] = 50
kwargs_dict_reacher_easy_dmp['controller_kwargs']['d_gains'] = 1
register(
    id=f'dmc_reacher-easy_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "traj_gen_kwargs": {
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

kwargs_dict_reacher_easy_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_reacher_easy_promp['name'] = f"reacher-easy"
kwargs_dict_reacher_easy_promp['wrappers'].append(suite.reacher.MPWrapper)
kwargs_dict_reacher_easy_promp['controller_kwargs']['p_gains'] = 50
kwargs_dict_reacher_easy_promp['controller_kwargs']['d_gains'] = 1
kwargs_dict_reacher_easy_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
register(
    id=f'dmc_reacher-easy_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("dmc_reacher-easy_promp-v0")

kwargs_dict_reacher_hard_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_reacher_hard_dmp['name'] = f"reacher-hard"
kwargs_dict_reacher_hard_dmp['wrappers'].append(suite.reacher.MPWrapper)
kwargs_dict_reacher_hard_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_reacher_hard_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
kwargs_dict_reacher_hard_dmp['controller_kwargs']['p_gains'] = 50
kwargs_dict_reacher_hard_dmp['controller_kwargs']['d_gains'] = 1
register(
    id=f'dmc_reacher-hard_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "traj_gen_kwargs": {
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

kwargs_dict_reacher_hard_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_reacher_hard_promp['name'] = f"reacher-hard"
kwargs_dict_reacher_hard_promp['wrappers'].append(suite.reacher.MPWrapper)
kwargs_dict_reacher_hard_promp['controller_kwargs']['p_gains'] = 50
kwargs_dict_reacher_hard_promp['controller_kwargs']['d_gains'] = 1
kwargs_dict_reacher_hard_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
register(
    id=f'dmc_reacher-hard_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 20,
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("dmc_reacher-hard_promp-v0")

_dmc_cartpole_tasks = ["balance", "balance_sparse", "swingup", "swingup_sparse"]

for _task in _dmc_cartpole_tasks:
    _env_id = f'dmc_cartpole-{_task}_dmp-v0'
    kwargs_dict_cartpole_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_cartpole_dmp['name'] = f"cartpole-{_task}"
    kwargs_dict_cartpole_dmp['camera_id'] = 0
    kwargs_dict_cartpole_dmp['wrappers'].append(suite.cartpole.MPWrapper)
    kwargs_dict_cartpole_dmp['phase_generator_kwargs']['alpha_phase'] = 2
    kwargs_dict_cartpole_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
    kwargs_dict_cartpole_dmp['controller_kwargs']['p_gains'] = 10
    kwargs_dict_cartpole_dmp['controller_kwargs']['d_gains'] = 10
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"cartpole-{_task}",
            # "time_limit": 1,
            "camera_id": 0,
            "episode_length": 1000,
            "wrappers": [suite.cartpole.MPWrapper],
            "traj_gen_kwargs": {
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
    ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'dmc_cartpole-{_task}_promp-v0'
    kwargs_dict_cartpole_promp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_cartpole_promp['name'] = f"cartpole-{_task}"
    kwargs_dict_cartpole_promp['camera_id'] = 0
    kwargs_dict_cartpole_promp['wrappers'].append(suite.cartpole.MPWrapper)
    kwargs_dict_cartpole_promp['controller_kwargs']['p_gains'] = 10
    kwargs_dict_cartpole_promp['controller_kwargs']['d_gains'] = 10
    kwargs_dict_cartpole_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"cartpole-{_task}",
            # "time_limit": 1,
            "camera_id": 0,
            "episode_length": 1000,
            "wrappers": [suite.cartpole.MPWrapper],
            "traj_gen_kwargs": {
                "num_dof": 1,
                "num_basis": 5,
                "duration": 10,
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
    ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

kwargs_dict_cartpole2poles_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_cartpole2poles_dmp['name'] = f"cartpole-two_poles"
kwargs_dict_cartpole2poles_dmp['camera_id'] = 0
kwargs_dict_cartpole2poles_dmp['wrappers'].append(suite.cartpole.TwoPolesMPWrapper)
kwargs_dict_cartpole2poles_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_cartpole2poles_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
kwargs_dict_cartpole2poles_dmp['controller_kwargs']['p_gains'] = 10
kwargs_dict_cartpole2poles_dmp['controller_kwargs']['d_gains'] = 10
_env_id = f'dmc_cartpole-two_poles_dmp-v0'
register(
    id=_env_id,
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-two_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        "wrappers": [suite.cartpole.TwoPolesMPWrapper],
        "traj_gen_kwargs": {
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

kwargs_dict_cartpole2poles_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_cartpole2poles_promp['name'] = f"cartpole-two_poles"
kwargs_dict_cartpole2poles_promp['camera_id'] = 0
kwargs_dict_cartpole2poles_promp['wrappers'].append(suite.cartpole.TwoPolesMPWrapper)
kwargs_dict_cartpole2poles_promp['controller_kwargs']['p_gains'] = 10
kwargs_dict_cartpole2poles_promp['controller_kwargs']['d_gains'] = 10
kwargs_dict_cartpole2poles_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
_env_id = f'dmc_cartpole-two_poles_promp-v0'
register(
    id=_env_id,
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"cartpole-two_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        "wrappers": [suite.cartpole.TwoPolesMPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 1,
            "num_basis": 5,
            "duration": 10,
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

kwargs_dict_cartpole3poles_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_cartpole3poles_dmp['name'] = f"cartpole-three_poles"
kwargs_dict_cartpole3poles_dmp['camera_id'] = 0
kwargs_dict_cartpole3poles_dmp['wrappers'].append(suite.cartpole.ThreePolesMPWrapper)
kwargs_dict_cartpole3poles_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_cartpole3poles_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
kwargs_dict_cartpole3poles_dmp['controller_kwargs']['p_gains'] = 10
kwargs_dict_cartpole3poles_dmp['controller_kwargs']['d_gains'] = 10
_env_id = f'dmc_cartpole-three_poles_dmp-v0'
register(
    id=_env_id,
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-three_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        "wrappers": [suite.cartpole.ThreePolesMPWrapper],
        "traj_gen_kwargs": {
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

kwargs_dict_cartpole3poles_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_cartpole3poles_promp['name'] = f"cartpole-three_poles"
kwargs_dict_cartpole3poles_promp['camera_id'] = 0
kwargs_dict_cartpole3poles_promp['wrappers'].append(suite.cartpole.ThreePolesMPWrapper)
kwargs_dict_cartpole3poles_promp['controller_kwargs']['p_gains'] = 10
kwargs_dict_cartpole3poles_promp['controller_kwargs']['d_gains'] = 10
kwargs_dict_cartpole3poles_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
_env_id = f'dmc_cartpole-three_poles_promp-v0'
register(
    id=_env_id,
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"cartpole-three_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        "wrappers": [suite.cartpole.ThreePolesMPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 1,
            "num_basis": 5,
            "duration": 10,
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

# DeepMind Manipulation
kwargs_dict_mani_reach_site_features_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_mani_reach_site_features_dmp['name'] = f"manipulation-reach_site_features"
kwargs_dict_mani_reach_site_features_dmp['wrappers'].append(manipulation.reach_site.MPWrapper)
kwargs_dict_mani_reach_site_features_dmp['phase_generator_kwargs']['alpha_phase'] = 2
kwargs_dict_mani_reach_site_features_dmp['trajectory_generator_kwargs']['weight_scale'] = 500  # TODO: weight scale 50, but goal scale 0.1
kwargs_dict_mani_reach_site_features_dmp['controller_kwargs']['controller_type'] = 'velocity'
register(
    id=f'dmc_manipulation-reach_site_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [manipulation.reach_site.MPWrapper],
        "traj_gen_kwargs": {
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

kwargs_dict_mani_reach_site_features_promp = deepcopy(DEFAULT_BB_DICT_DMP)
kwargs_dict_mani_reach_site_features_promp['name'] = f"manipulation-reach_site_features"
kwargs_dict_mani_reach_site_features_promp['wrappers'].append(manipulation.reach_site.MPWrapper)
kwargs_dict_mani_reach_site_features_promp['trajectory_generator_kwargs']['weight_scale'] = 0.2
kwargs_dict_mani_reach_site_features_promp['controller_kwargs']['controller_type'] = 'velocity'
register(
    id=f'dmc_manipulation-reach_site_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [manipulation.reach_site.MPWrapper],
        "traj_gen_kwargs": {
            "num_dof": 9,
            "num_basis": 5,
            "duration": 10,
            "policy_type": "velocity",
            "weights_scale": 0.2,
            "zero_start": True,
        }
    }
)
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"].append("dmc_manipulation-reach_site_promp-v0")
