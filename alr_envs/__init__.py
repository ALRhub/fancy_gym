from alr_envs import dmc, meta, open_ai
from alr_envs.utils.make_env_helpers import make, make_bb, make_rank

# Convenience function for all MP environments
from .envs import ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS
from .dmc import ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS
from .meta import ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS
from .open_ai import ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS

ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS = {
    key: value + ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS[key]
    for key, value in ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS.items()}
