from alr_envs import dmc, meta, open_ai
from alr_envs.utils.make_env_helpers import make, make_detpmp_env, make_dmp_env, make_rank
from alr_envs.utils import make_dmc

# Convenience function for all MP environments
from .alr import ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS
from .dmc import ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS
from .meta import ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS
from .open_ai import ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS

ALL_MOTION_PRIMITIVE_ENVIRONMENTS = {
    key: value + ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS[key] +
         ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS[key]
    for key, value in ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS.items()}
