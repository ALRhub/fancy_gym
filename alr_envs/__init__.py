from gym.envs.registration import register

from alr_envs.stochastic_search.functions.f_rosenbrock import Rosenbrock
from alr_envs.utils.wrapper.dmp_wrapper import DmpWrapper

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
        "hole_width": 0.15,
        "hole_depth": 1,
        "hole_x": 1,
        "collision_penalty": 100,
    }
)

# DMP environments

register(
    id='ViaPointReacherDMP-v0',
    entry_point='alr_envs.classic_control.viapoint_reacher:viapoint_dmp',
    # max_episode_steps=1,
)

register(
    id='HoleReacherDMP-v0',
    entry_point='alr_envs.classic_control.hole_reacher:holereacher_dmp',
    # max_episode_steps=1,
)

register(
    id='HoleReacherFixedGoalDMP-v0',
    entry_point='alr_envs.classic_control.hole_reacher:holereacher_fix_goal_dmp',
    # max_episode_steps=1,
)

register(
    id='HoleReacherDetPMP-v0',
    entry_point='alr_envs.classic_control.hole_reacher:holereacher_detpmp',
    # max_episode_steps=1,
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
