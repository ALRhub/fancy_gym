from . import manipulation, suite

ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS = {"DMP": [], "ProMP": []}

from gym.envs.registration import register

# DeepMind Control Suite (DMC)

register(
    id=f'dmc_ball_in_cup-catch_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.ball_in_cup.MPWrapper],
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
    id=f'dmc_ball_in_cup-catch_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.ball_in_cup.MPWrapper],
        "mp_kwargs": {
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

register(
    id=f'dmc_reacher-easy_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
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
    id=f'dmc_reacher-easy_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "mp_kwargs": {
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

register(
    id=f'dmc_reacher-hard_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
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
    id=f'dmc_reacher-hard_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 20,
        "episode_length": 1000,
        "wrappers": [suite.reacher.MPWrapper],
        "mp_kwargs": {
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
    ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'dmc_cartpole-{_task}_promp-v0'
    register(
        id=_env_id,
        entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
        kwargs={
            "name": f"cartpole-{_task}",
            # "time_limit": 1,
            "camera_id": 0,
            "episode_length": 1000,
            "wrappers": [suite.cartpole.MPWrapper],
            "mp_kwargs": {
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

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
        "mp_kwargs": {
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
ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

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
        "mp_kwargs": {
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

register(
    id=f'dmc_manipulation-reach_site_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [manipulation.reach_site.MPWrapper],
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
    id=f'dmc_manipulation-reach_site_promp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_promp_env_helper',
    kwargs={
        "name": f"manipulation-reach_site_features",
        # "time_limit": 1,
        "episode_length": 250,
        "wrappers": [manipulation.reach_site.MPWrapper],
        "mp_kwargs": {
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
