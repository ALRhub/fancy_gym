from .reacher.balancing import BalancingEnv
from .ball_in_a_cup.ball_in_a_cup import ALRBallInACupEnv
from .ball_in_a_cup.biac_pd import ALRBallInACupPDEnv
from .table_tennis.tt_gym import TTEnvGym
from .beerpong.beerpong import ALRBeerBongEnv
from .ant_jump.ant_jump import ALRAntJumpEnv
from .half_cheetah_jump.half_cheetah_jump import ALRHalfCheetahJumpEnv
from .hopper_jump.hopper_jump import ALRHopperJumpEnv, ALRHopperJumpRndmPosEnv, ALRHopperXYJumpEnv
from .hopper_jump.hopper_jump_on_box import ALRHopperJumpOnBoxEnv
from .hopper_throw.hopper_throw import ALRHopperThrowEnv
from .hopper_throw.hopper_throw_in_basket import ALRHopperThrowInBasketEnv
from .walker_2d_jump.walker_2d_jump import ALRWalker2dJumpEnv
from .reacher.alr_reacher import ALRReacherEnv