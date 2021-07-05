import numpy as np
from gym.envs.registration import register

from alr_envs.classic_control.hole_reacher.hole_reacher_mp_wrapper import HoleReacherMPWrapper
from alr_envs.classic_control.simple_reacher.simple_reacher_mp_wrapper import SimpleReacherMPWrapper
from alr_envs.classic_control.viapoint_reacher.viapoint_reacher_mp_wrapper import ViaPointReacherMPWrapper
from alr_envs.dmc.ball_in_cup.ball_in_the_cup_mp_wrapper import DMCBallInCupMPWrapper
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_mp_wrapper import BallInACupMPWrapper
from alr_envs.open_ai import reacher_v2, continuous_mountain_car, fetch
from alr_envs.stochastic_search.functions.f_rosenbrock import Rosenbrock

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

register(
    id='ALRBallInACupSimple-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupEnv',
    max_episode_steps=4000,
    kwargs={
        "simplified": True,
        "reward_type": "no_context",
    }
)

register(
    id='ALRBallInACupPDSimple-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupPDEnv',
    max_episode_steps=4000,
    kwargs={
        "simplified": True,
        "reward_type": "no_context"
    }
)

register(
    id='ALRBallInACupPD-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupPDEnv',
    max_episode_steps=4000,
    kwargs={
        "simplified": False,
        "reward_type": "no_context"
    }
)

register(
    id='ALRBallInACup-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupEnv',
    max_episode_steps=4000,
    kwargs={
        "reward_type": "no_context"
    }
)

register(
    id='ALRBallInACupGoal-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupEnv',
    max_episode_steps=4000,
    kwargs={
        "reward_type": "contextual_goal"
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
        "hole_width": None,
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

# MP environments

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
            "wrappers": [SimpleReacherMPWrapper],
            "mp_kwargs": {
                "num_dof": 2 if "long" not in v.lower() else 5,
                "num_basis": 5,
                "duration": 2,
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
        "wrappers": [ViaPointReacherMPWrapper],
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

## Hole Reacher
versions = ["v0", "v1", "v2"]
for v in versions:
    register(
        id=f'HoleReacherDMP-{v}',
        entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
        # max_episode_steps=1,
        kwargs={
            "name": f"alr_envs:HoleReacher-{v}",
            "wrappers": [HoleReacherMPWrapper],
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
            "wrappers": [HoleReacherMPWrapper],
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

# TODO: properly add final_pos
register(
    id='HoleReacherFixedGoalDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:HoleReacher-v0",
        "wrappers": [HoleReacherMPWrapper],
        "mp_kwargs": {
            "num_dof": 5,
            "num_basis": 5,
            "duration": 2,
            "learn_goal": False,
            "alpha_phase": 2,
            "policy_type": "velocity",
            "weights_scale": 50,
            "goal_scale": 0.1
        }
    }
)

## Ball in Cup

register(
    id='ALRBallInACupSimpleDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    kwargs={
        "name": "alr_envs:ALRBallInACupSimple-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 3,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "learn_goal": False,
            "alpha_phase": 3,
            "bandwidth_factor": 2.5,
            "policy_type": "motor",
            "weights_scale": 100,
            "return_to_start": True,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

register(
    id='ALRBallInACupDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    kwargs={
        "name": "alr_envs:ALRBallInACup-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 7,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "learn_goal": False,
            "alpha_phase": 3,
            "bandwidth_factor": 2.5,
            "policy_type": "motor",
            "weights_scale": 100,
            "return_to_start": True,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

register(
    id='ALRBallInACupSimpleDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "alr_envs:ALRBallInACupSimple-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 3,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "width": 0.0035,
            # "off": -0.05,
            "policy_type": "motor",
            "weights_scale": 0.2,
            "zero_start": True,
            "zero_goal": True,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

register(
    id='ALRBallInACupPDSimpleDetPMP-v0',
    entry_point='alr_envs.mujoco.ball_in_a_cup.biac_pd:make_detpmp_env_helper',
    kwargs={
        "name": "alr_envs:ALRBallInACupPDSimple-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 3,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "width": 0.0035,
            # "off": -0.05,
            "policy_type": "motor",
            "weights_scale": 0.2,
            "zero_start": True,
            "zero_goal": True,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

register(
    id='ALRBallInACupPDDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env',
    kwargs={
        "name": "alr_envs:ALRBallInACupPD-v0",
        "num_dof": 7,
        "num_basis": 5,
        "duration": 3.5,
        "post_traj_time": 4.5,
        "width": 0.0035,
        # "off": -0.05,
        "policy_type": "motor",
        "weights_scale": 0.2,
        "zero_start": True,
        "zero_goal": True,
        "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
        "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
    }
)

register(
    id='ALRBallInACupDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "alr_envs:ALRBallInACupSimple-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 7,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "width": 0.0035,
            "policy_type": "motor",
            "weights_scale": 0.2,
            "zero_start": True,
            "zero_goal": True,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

register(
    id='ALRBallInACupGoalDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_contextual_env',
    kwargs={
        "name": "alr_envs:ALRBallInACupGoal-v0",
        "wrappers": [BallInACupMPWrapper],
        "mp_kwargs": {
            "num_dof": 7,
            "num_basis": 5,
            "duration": 3.5,
            "post_traj_time": 4.5,
            "learn_goal": True,
            "alpha_phase": 3,
            "bandwidth_factor": 2.5,
            "policy_type": "motor",
            "weights_scale": 50,
            "goal_scale": 0.1,
            "policy_kwargs": {
                "p_gains": np.array([4. / 3., 2.4, 2.5, 5. / 3., 2., 2., 1.25]),
                "d_gains": np.array([0.0466, 0.12, 0.125, 0.04166, 0.06, 0.06, 0.025])
            }
        }
    }
)

## DMC

register(
    id=f'dmc_ball_in_cup-catch_dmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env_helper',
    # max_episode_steps=1,
    kwargs={
        "name": f"ball_in_cup-catch",
        "wrappers": [DMCBallInCupMPWrapper],
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

register(
    id=f'dmc_ball_in_cup-catch_detpmp-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": f"ball_in_cup-catch",
        "wrappers": [DMCBallInCupMPWrapper],
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

## Open AI
register(
    id='ContinuousMountainCarDetPMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_detpmp_env_helper',
    kwargs={
        "name": "gym.envs.classic_control:MountainCarContinuous-v0",
        "wrappers": [continuous_mountain_car.PositionalWrapper, continuous_mountain_car.MPWrapper],
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
        "wrappers": [reacher_v2.PositionalWrapper, reacher_v2.MPWrapper],
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
        "wrappers": [fetch.PositionalWrapper, fetch.MPWrapper],
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
        "wrappers": [fetch.PositionalWrapper, fetch.MPWrapper],
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

# BBO functions

for dim in [5, 10, 25, 50, 100]:
    register(
        id=f'Rosenbrock{dim}-v0',
        entry_point='alr_envs.stochastic_search:StochasticSearchEnv',
        max_episode_steps=1,
        kwargs={
            "cost_f": Rosenbrock(dim),
        }
    )
