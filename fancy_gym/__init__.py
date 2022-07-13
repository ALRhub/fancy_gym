from fancy_gym import dmc, meta, open_ai
from fancy_gym.utils.make_env_helpers import make, make_bb, make_rank
from .dmc import ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS
# Convenience function for all MP environments
from .envs import ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS
from .meta import ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS
from .open_ai import ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS

ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {
    key: value + ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS[key]
    for key, value in ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS.items()}
