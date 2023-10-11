from copy import deepcopy

import numpy as np
from gymnasium import register as gym_register
from .registry import register, upgrade

from . import classic_control, mujoco
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.simple_reacher import MPWrapper as MPWrapper_SimpleReacher
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.hole_reacher import MPWrapper as MPWrapper_HoleReacher
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .classic_control.viapoint_reacher import MPWrapper as MPWrapper_ViaPointReacher
from .mujoco.reacher.reacher import ReacherEnv, MAX_EPISODE_STEPS_REACHER
from .mujoco.reacher.mp_wrapper import MPWrapper as MPWrapper_Reacher
from .mujoco.ant_jump.ant_jump import MAX_EPISODE_STEPS_ANTJUMP
from .mujoco.beerpong.beerpong import MAX_EPISODE_STEPS_BEERPONG, FIXED_RELEASE_STEP
from .mujoco.beerpong.mp_wrapper import MPWrapper as MPWrapper_Beerpong
from .mujoco.beerpong.mp_wrapper import MPWrapper_FixedRelease as MPWrapper_Beerpong_FixedRelease
from .mujoco.half_cheetah_jump.half_cheetah_jump import MAX_EPISODE_STEPS_HALFCHEETAHJUMP
from .mujoco.hopper_jump.hopper_jump import MAX_EPISODE_STEPS_HOPPERJUMP
from .mujoco.hopper_jump.hopper_jump_on_box import MAX_EPISODE_STEPS_HOPPERJUMPONBOX
from .mujoco.hopper_throw.hopper_throw import MAX_EPISODE_STEPS_HOPPERTHROW
from .mujoco.hopper_throw.hopper_throw_in_basket import MAX_EPISODE_STEPS_HOPPERTHROWINBASKET
from .mujoco.walker_2d_jump.walker_2d_jump import MAX_EPISODE_STEPS_WALKERJUMP
from .mujoco.box_pushing.box_pushing_env import BoxPushingDense, BoxPushingTemporalSparse, \
    BoxPushingTemporalSpatialSparse, MAX_EPISODE_STEPS_BOX_PUSHING
from .mujoco.table_tennis.table_tennis_env import TableTennisEnv, TableTennisWind, TableTennisGoalSwitching, \
    MAX_EPISODE_STEPS_TABLE_TENNIS
from .mujoco.table_tennis.mp_wrapper import TT_MPWrapper as MPWrapper_TableTennis
from .mujoco.table_tennis.mp_wrapper import TT_MPWrapper_Replan as MPWrapper_TableTennis_Replan
from .mujoco.table_tennis.mp_wrapper import TTVelObs_MPWrapper as MPWrapper_TableTennis_VelObs
from .mujoco.table_tennis.mp_wrapper import TTVelObs_MPWrapper_Replan as MPWrapper_TableTennis_VelObs_Replan

# Classic Control
# Simple Reacher
register(
    id='fancy/SimpleReacher-v0',
    entry_point=SimpleReacherEnv,
    mp_wrapper=MPWrapper_SimpleReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='fancy/LongSimpleReacher-v0',
    entry_point=SimpleReacherEnv,
    mp_wrapper=MPWrapper_SimpleReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

# Viapoint Reacher
register(
    id='fancy/ViaPointReacher-v0',
    entry_point=ViaPointReacherEnv,
    mp_wrapper=MPWrapper_ViaPointReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "collision_penalty": 1000
    }
)

# Hole Reacher
register(
    id='fancy/HoleReacher-v0',
    entry_point=HoleReacherEnv,
    mp_wrapper=MPWrapper_HoleReacher,
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

# Mujoco

# Mujoco Reacher
for dims in [5, 7]:
    register(
        id=f'fancy/Reacher{dims}d-v0',
        entry_point=ReacherEnv,
        mp_wrapper=MPWrapper_Reacher,
        max_episode_steps=MAX_EPISODE_STEPS_REACHER,
        kwargs={
            "n_links": dims,
        }
    )

    register(
        id=f'fancy/Reacher{dims}dSparse-v0',
        entry_point=ReacherEnv,
        mp_wrapper=MPWrapper_Reacher,
        max_episode_steps=MAX_EPISODE_STEPS_REACHER,
        kwargs={
            "sparse": True,
            'reward_weight': 200,
            "n_links": dims,
        }
    )


register(
    id='fancy/HopperJumpSparse-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpEnv',
    mp_wrapper=mujoco.hopper_jump.MPWrapper,
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": True,
    }
)

register(
    id='fancy/HopperJump-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpEnv',
    mp_wrapper=mujoco.hopper_jump.MPWrapper,
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": False,
        "healthy_reward": 1.0,
        "contact_weight": 0.0,
        "height_weight": 3.0,
    }
)

# TODO: Add [MPs] later when finished (old TODO I moved here during refactor)
register(
    id='fancy/AntJump-v0',
    entry_point='fancy_gym.envs.mujoco:AntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    add_mp_types=[],
)

register(
    id='fancy/HalfCheetahJump-v0',
    entry_point='fancy_gym.envs.mujoco:HalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    add_mp_types=[],
)

register(
    id='fancy/HopperJumpOnBox-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
    add_mp_types=[],
)

register(
    id='fancy/HopperThrow-v0',
    entry_point='fancy_gym.envs.mujoco:HopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    add_mp_types=[],
)

register(
    id='fancy/HopperThrowInBasket-v0',
    entry_point='fancy_gym.envs.mujoco:HopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    add_mp_types=[],
)

register(
    id='fancy/Walker2DJump-v0',
    entry_point='fancy_gym.envs.mujoco:Walker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    add_mp_types=[],
)

register(  # [MPDone
    id='fancy/BeerPong-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnv',
    mp_wrapper=MPWrapper_Beerpong,
    max_episode_steps=MAX_EPISODE_STEPS_BEERPONG,
    add_mp_types=['ProMP'],
)

# Here we use the same reward as in BeerPong-v0, but now consider after the release,
# only one time step, i.e. we simulate until the end of th episode
register(
    id='fancy/BeerPongStepBased-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnvStepBasedEpisodicReward',
    mp_wrapper=MPWrapper_Beerpong_FixedRelease,
    max_episode_steps=FIXED_RELEASE_STEP,
    add_mp_types=['ProMP'],
)

register(
    id='fancy/BeerPongFixedRelease-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnv',
    mp_wrapper=MPWrapper_Beerpong_FixedRelease,
    max_episode_steps=FIXED_RELEASE_STEP,
    add_mp_types=['ProMP'],
)

# Box pushing environments with different rewards
for reward_type in ["Dense", "TemporalSparse", "TemporalSpatialSparse"]:
    register(
        id='fancy/BoxPushing{}-v0'.format(reward_type),
        entry_point='fancy_gym.envs.mujoco:BoxPushing{}'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.MPWrapper,
        max_episode_steps=MAX_EPISODE_STEPS_BOX_PUSHING,
    )
    register(
        id='fancy/BoxPushingRandomInit{}-v0'.format(reward_type),
        entry_point='fancy_gym.envs.mujoco:BoxPushing{}'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.MPWrapper,
        max_episode_steps=MAX_EPISODE_STEPS_BOX_PUSHING,
        kwargs={"random_init": True}
    )

    upgrade(
        id='fancy/BoxPushing{}Replan-v0'.format(reward_type),
        base_id='fancy/BoxPushing{}-v0'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.ReplanMPWrapper,
    )

# Table Tennis environments
for ctxt_dim in [2, 4]:
    register(
        id='fancy/TableTennis{}D-v0'.format(ctxt_dim),
        entry_point='fancy_gym.envs.mujoco:TableTennisEnv',
        mp_wrapper=MPWrapper_TableTennis,
        max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
        add_mp_types=['ProMP', 'ProDMP'],
        kwargs={
            "ctxt_dim": ctxt_dim,
            'frame_skip': 4,
        }
    )

    register(
        id='fancy/TableTennis{}DReplan-v0'.format(ctxt_dim),
        entry_point='fancy_gym.envs.mujoco:TableTennisEnv',
        mp_wrapper=MPWrapper_TableTennis,
        max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
        add_mp_types=['ProDMP'],
        kwargs={
            "ctxt_dim": ctxt_dim,
            'frame_skip': 4,
        }
    )

register(
    id='fancy/TableTennisWind-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisWind',
    mp_wrapper=MPWrapper_TableTennis_VelObs,
    add_mp_types=['ProMP', 'ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
)

register(
    id='fancy/TableTennisWindReplan-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisWind',
    mp_wrapper=MPWrapper_TableTennis_VelObs_Replan,
    add_mp_types=['ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
)

register(
    id='fancy/TableTennisGoalSwitching-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisGoalSwitching',
    mp_wrapper=MPWrapper_TableTennis,
    add_mp_types=['ProMP', 'ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
    kwargs={
        'goal_switching_step': 99
    }
)

register(
    id='fancy/TableTennisGoalSwitchingReplan-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisGoalSwitching',
    mp_wrapper=MPWrapper_TableTennis_Replan,
    add_mp_types=['ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
    kwargs={
        'goal_switching_step': 99
    }
)
