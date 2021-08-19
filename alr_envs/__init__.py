from gym.envs.registration import register
from gym.wrappers import FlattenObservation

from alr_envs import classic_control, dmc, open_ai, meta

from alr_envs.utils.make_env_helpers import make_dmp_env
from alr_envs.utils.make_env_helpers import make_detpmp_env
from alr_envs.utils.make_env_helpers import make
from alr_envs.utils.make_env_helpers import make_rank

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
    register(
        id=f'{name[0]}DMP-{name[1]}',
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

## Hole Reacher
versions = ["v0", "v1", "v2"]
for v in versions:
    register(
        id=f'HoleReacherDMP-{v}',
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

    register(
        id=f'HoleReacherDetPMP-{v}',
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

## Deep Mind Control Suite (DMC)
### Suite

register(
    id=f'dmc_ball_in_cup-catch_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 2,
        "episode_length": 100,
        "wrappers": [dmc.suite.ball_in_cup.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 2,
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

register(
    id=f'dmc_ball_in_cup-catch_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"ball_in_cup-catch",
        "time_limit": 2,
        "episode_length": 100,
        "wrappers": [dmc.suite.ball_in_cup.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 2,
            "width": 0.025,
            "policy_type": "motor",
            "weights_scale": 1,
            "zero_start": True,
            "policy_kwargs": {
                "p_gains": 50,
                "d_gains": 1
            }
        }
    }
)

# TODO tune episode length for all below
register(
    id=f'dmc_reacher-easy_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 1,
        "episode_length": 50,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 1,
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

register(
    id=f'dmc_reacher-easy_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"reacher-easy",
        "time_limit": 1,
        "episode_length": 50,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 1,
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

register(
    id=f'dmc_reacher-hard_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 1,
        "episode_length": 50,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 1,
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

register(
    id=f'dmc_reacher-hard_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"reacher-hard",
        "time_limit": 1,
        "episode_length": 50,
        "wrappers": [dmc.suite.reacher.MPWrapper],
        "mp_kwargs": {
            "num_dof": 2,
            "num_basis": 5,
            "duration": 1,
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
register(
    id=f'dmc_cartpole-balance_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-balance",
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

register(
    id=f'dmc_cartpole-balance_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-balance",
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
register(
    id=f'dmc_cartpole-balance_sparse_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-balance_sparse",
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

register(
    id=f'dmc_cartpole-balance_sparse_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-balance_sparse",
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

register(
    id=f'dmc_cartpole-swingup_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-swingup",
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

register(
    id=f'dmc_cartpole-swingup_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-swingup",
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
register(
    id=f'dmc_cartpole-swingup_sparse_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-swingup_sparse",
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

register(
    id=f'dmc_cartpole-swingup_sparse_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-swingup_sparse",
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
register(
    id=f'dmc_cartpole-two_poles_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-two_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        # "wrappers": [partial(DMCCartpoleMPWrapper, n_poles=2)],
        "wrappers": [dmc.suite.cartpole.TwoPolesMPWrapper],
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

register(
    id=f'dmc_cartpole-two_poles_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-two_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        # "wrappers": [partial(DMCCartpoleMPWrapper, n_poles=2)],
        "wrappers": [dmc.suite.cartpole.TwoPolesMPWrapper],
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
register(
    id=f'dmc_cartpole-three_poles_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"cartpole-three_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        # "wrappers": [partial(DMCCartpoleMPWrapper, n_poles=3)],
        "wrappers": [dmc.suite.cartpole.ThreePolesMPWrapper],
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

register(
    id=f'dmc_cartpole-three_poles_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"cartpole-three_poles",
        # "time_limit": 1,
        "camera_id": 0,
        "episode_length": 1000,
        # "wrappers": [partial(DMCCartpoleMPWrapper, n_poles=3)],
        "wrappers": [dmc.suite.cartpole.ThreePolesMPWrapper],
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
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": 1.,
                "d_gains": 1.
            }
        }
    }
)

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
            "policy_type": "motor",
            "policy_kwargs": {
                "p_gains": .6,
                "d_gains": .075
            }
        }
    }
)

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
            "policy_type": "position"
        }
    }
)

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
            "policy_type": "position"
        }
    }
)

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
            "policy_type": "position"
        }
    }
)

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
            "policy_type": "position"
        }
    }
)

register(
    id='ButtonPressDetPMP-v2',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "button-press-v2",
        "wrappers": [meta.button_press.MPWrapper],
        "mp_kwargs": {
            "num_dof": 4,
            "num_basis": 5,
            "duration": 6.25,
            "post_traj_time": 0,
            "width": 0.025,
            "policy_type": "position"
        }
    }
)

