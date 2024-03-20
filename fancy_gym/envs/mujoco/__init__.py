from .ant_jump.ant_jump import AntJumpEnv
from .beerpong.beerpong import BeerPongEnv, BeerPongEnvStepBasedEpisodicReward
from .half_cheetah_jump.half_cheetah_jump import HalfCheetahJumpEnv
from .hopper_jump.hopper_jump import HopperJumpEnv, HopperJumpMarkovRew
from .hopper_jump.hopper_jump_on_box import HopperJumpOnBoxEnv
from .hopper_throw.hopper_throw import HopperThrowEnv
from .hopper_throw.hopper_throw_in_basket import HopperThrowInBasketEnv
from .reacher.reacher import ReacherEnv
from .walker_2d_jump.walker_2d_jump import Walker2dJumpEnv
from .box_pushing.box_pushing_env import BoxPushingDense, BoxPushingTemporalSparse, BoxPushingTemporalSpatialSparse
from .table_tennis.table_tennis_env import TableTennisEnv, TableTennisWind, TableTennisGoalSwitching, TableTennisMarkov, TableTennisRandomInit

try:
    from .air_hockey.air_hockey_env_wrapper import AirHockeyEnv
except ModuleNotFoundError:
    print("[FANCY GYM] Air Hockey not available (depends on mushroom-rl, dmc, mujoco)")