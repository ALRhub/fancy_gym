from gym.envs.registration import register

from alr_envs.stochastic_search.functions.f_rosenbrock import Rosenbrock
# from alr_envs.utils.wrapper.dmp_wrapper import DmpWrapper

# Mujoco

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

register(
    id='ALRBallInACupSimple-v0',
    entry_point='alr_envs.mujoco:ALRBallInACupEnv',
    max_episode_steps=4000,
    kwargs={
        "simplified": True,
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

register(
    id='Balancing-v0',
    entry_point='alr_envs.mujoco:BalancingEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

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
    id='ViaPointReacher-v0',
    entry_point='alr_envs.classic_control.viapoint_reacher:ViaPointReacher',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "collision_penalty": 1000
    }
)

register(
    id='HoleReacher-v0',
    entry_point='alr_envs.classic_control.hole_reacher:HoleReacher',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": 0.25,
        "hole_depth": 1,
        "hole_x": 2,
        "collision_penalty": 100,
    }
)

register(
    id='HoleReacher-v2',
    entry_point='alr_envs.classic_control.hole_reacher_v2:HoleReacher',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": 0.25,
        "hole_depth": 1,
        "hole_x": 2,
        "collision_penalty": 100,
    }
)

# MP environments

register(
    id='SimpleReacherDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:SimpleReacher-v0",
        "num_dof": 2,
        "num_basis": 5,
        "duration": 2,
        "alpha_phase": 2,
        "learn_goal": True,
        "policy_type": "velocity",
        "weights_scale": 50,
    }
)

register(
    id='SimpleReacherDMP-v1',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:SimpleReacher-v1",
        "num_dof": 2,
        "num_basis": 5,
        "duration": 2,
        "alpha_phase": 2,
        "learn_goal": True,
        "policy_type": "velocity",
        "weights_scale": 50,
    }
)

register(
    id='ViaPointReacherDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:ViaPointReacher-v0",
        "num_dof": 5,
        "num_basis": 5,
        "duration": 2,
        "alpha_phase": 2,
        "learn_goal": False,
        "policy_type": "velocity",
        "weights_scale": 50,
    }
)

register(
    id='HoleReacherDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:HoleReacher-v0",
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
)

register(
    id='HoleReacherDMP-v2',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:HoleReacher-v2",
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
)

# TODO: properly add final_pos
register(
    id='HoleReacherFixedGoalDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    # max_episode_steps=1,
    kwargs={
        "name": "alr_envs:HoleReacher-v0",
        "num_dof": 5,
        "num_basis": 5,
        "duration": 2,
        "learn_goal": False,
        "alpha_phase": 2,
        "policy_type": "velocity",
        "weights_scale": 50,
        "goal_scale": 0.1
    }
)

# register(
#     id='HoleReacherDetPMP-v0',
#     entry_point='alr_envs.classic_control.hole_reacher:holereacher_detpmp',
#     # max_episode_steps=1,
#     # TODO: add mp kwargs
# )

register(
    id='ALRBallInACupSimpleDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    kwargs={
        "name": "alr_envs:ALRBallInACupSimple-v0",
        "num_dof": 3,
        "num_basis": 5,
        "duration": 3.5,
        "post_traj_time": 4.5,
        "learn_goal": False,
        "alpha_phase": 3,
        "bandwidth_factor": 2.5,
        "policy_type": "motor",
        "weights_scale": 100,
        "return_to_start": True
    }
)

register(
    id='ALRBallInACupDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_dmp_env',
    kwargs={
        "name": "alr_envs:ALRBallInACup-v0",
        "num_dof": 7,
        "num_basis": 5,
        "duration": 3.5,
        "post_traj_time": 4.5,
        "learn_goal": False,
        "alpha_phase": 3,
        "bandwidth_factor": 2.5,
        "policy_type": "motor",
        "weights_scale": 100,
        "return_to_start": True
    }
)

register(
    id='ALRBallInACupGoalDMP-v0',
    entry_point='alr_envs.utils.make_env_helpers:make_contextual_env',
    kwargs={
        "name": "alr_envs:ALRBallInACupGoal-v0",
        "num_dof": 7,
        "num_basis": 5,
        "duration": 3.5,
        "post_traj_time": 4.5,
        "learn_goal": True,
        "alpha_phase": 3,
        "bandwidth_factor": 2.5,
        "policy_type": "motor",
        "weights_scale": 50,
        "goal_scale": 0.1
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
