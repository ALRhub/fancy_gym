from .ant_jump.ant_jump import AntJumpEnv
from .beerpong.beerpong import BeerPongEnv, BeerPongEnvStepBasedEpisodicReward
from .half_cheetah_jump.half_cheetah_jump import HalfCheetahJumpEnv
from .hopper_jump.hopper_jump import HopperJumpEnv
from .hopper_jump.hopper_jump_on_box import HopperJumpOnBoxEnv
from .hopper_throw.hopper_throw import HopperThrowEnv
from .hopper_throw.hopper_throw_in_basket import HopperThrowInBasketEnv
from .reacher.reacher import ReacherEnv
from .walker_2d_jump.walker_2d_jump import Walker2dJumpEnv
from .box_pushing.box_pushing_env import (BoxPushingDense, BoxPushingDenseRotInv, BoxPushingTemporalSparse,
                                          BoxPushingTemporalSparseRotInv, BoxPushingTemporalSpatialSparse,
                                          BoxPushingTemporalSpatialSparseRotInv,
                                          BoxPushingTemporalSparseNoGuidanceRotInv, BoxPushingTemporalSparseNoGuidanceAtAllRotInv,
                                          BoxPushingTemporalSparseNotInclinedInit)
from .box_pushing.box_pushing_obstacle_env import (BoxPushingObstacleDense, BoxPushingObstacleTemporalSparse,
                                                   BoxPushingObstacleTemporalSpatialSparse)
from .box_pushing.box_pushing_random2random_env import BoxPushingDenseRnd2Rnd, BoxPushingTemporalSparseRnd2Rnd
from .table_tennis.table_tennis_env import (TableTennisEnv, TableTennisWind, TableTennisGoalSwitching,
                                            TableTennisVelocity)
from .mini_golf.mini_golf_env import MiniGolfEnv, MiniGolfQuadRewEnv, MiniGolfOneObsEnv
