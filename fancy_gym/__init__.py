from fancy_gym import dmc, meta, open_ai
from fancy_gym import envs as fancy
from fancy_gym.utils.make_env_helpers import make_bb
from .envs.registry import register, upgrade
from .envs.registry import ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS, MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS

ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS = MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS['dm_control']
ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS = MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS['fancy']
if 'metaworld' in MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS:
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS = MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS['metaworld']
else:
    ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS = 'Metaworld is not installed.'
ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS = MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS['gym']


def make(*args, **kwargs):
    """
    As part of the refactor of Fancy Gym and upgrade to gymnasium the use of fancy_gym.make has been discontinued. Regular gym.make should be used instead. For more details check out the github README. If your codebase was build for older versions of Fancy Gym and relies on the old behavior and dependency versions, please check out the legacy branch.
    """
    raise Exception('As part of the refactor of Fancy Gym and upgrade to gymnasium the use of fancy_gym.make has been discontinued. Regular gym.make should be used instead. For more details check out the github README. If your codebase was build for older versions of Fancy Gym and relies on the old behavior and dependency versions, please check out the legacy branch.')
