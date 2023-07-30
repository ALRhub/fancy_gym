from fancy_gym.black_box.controller.meta_world_controller import MetaWorldController
from fancy_gym.black_box.controller.pd_controller import PDController
from fancy_gym.black_box.controller.pos_controller import PosController
from fancy_gym.black_box.controller.vel_controller import VelController

ALL_TYPES = ["motor", "velocity", "position", "metaworld"]


def get_controller(controller_type: str, **kwargs):
    controller_type = controller_type.lower()
    if controller_type == "motor":
        return PDController(**kwargs)
    elif controller_type == "velocity":
        return VelController(**kwargs)
    elif controller_type == "position":
        return PosController(**kwargs)
    elif controller_type == "metaworld":
        return MetaWorldController(**kwargs)
    else:
        raise ValueError(f"Specified controller type {controller_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
