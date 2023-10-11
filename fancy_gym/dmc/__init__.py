from copy import deepcopy

from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.registration import register

from ..envs.registry import register

from . import manipulation, suite

# DeepMind Control Suite (DMC)
register(
    id=f"dm_control/ball_in_cup-catch-v0",
    register_step_based=False,
    mp_wrapper=suite.ball_in_cup.MPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)

register(
    id=f"dm_control/reacher-easy-v0",
    register_step_based=False,
    mp_wrapper=suite.reacher.MPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)

register(
    id=f"dm_control/reacher-hard-v0",
    register_step_based=False,
    mp_wrapper=suite.reacher.MPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)

_dmc_cartpole_tasks = ["balance", "balance_sparse", "swingup", "swingup_sparse"]
for _task in _dmc_cartpole_tasks:
    register(
        id=f'dm_control/cartpole-{_task}-v0',
        register_step_based=False,
        mp_wrapper=suite.cartpole.MPWrapper,
        add_mp_types=['DMP', 'ProMP'],
    )

register(
    id=f"dm_control/cartpole-two_poles-v0",
    register_step_based=False,
    mp_wrapper=suite.cartpole.TwoPolesMPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)

register(
    id=f"dm_control/cartpole-three_poles-v0",
    register_step_based=False,
    mp_wrapper=suite.cartpole.ThreePolesMPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)

# DeepMind Manipulation
register(
    id=f"dm_control/reach_site_features-v0",
    register_step_based=False,
    mp_wrapper=manipulation.reach_site.MPWrapper,
    add_mp_types=['DMP', 'ProMP'],
)
