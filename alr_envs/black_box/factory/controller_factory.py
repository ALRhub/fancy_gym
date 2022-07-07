from alr_envs.black_box.controller.meta_world_controller import MetaWorldController
from alr_envs.black_box.controller.pd_controller import PDController
from alr_envs.black_box.controller.vel_controller import VelController
from alr_envs.black_box.controller.pos_controller import PosController

ALL_TYPES = ["motor", "velocity", "position", "metaworld"]


def get_controller(controller_type: str, **kwargs):
    controller_type = controller_type.lower()
    if controller_type == "motor":
        return PDController(**kwargs)
    elif controller_type == "velocity":
        return VelController()
    elif controller_type == "position":
        return PosController()
    elif controller_type == "metaworld":
        return MetaWorldController()
    else:
        raise ValueError(f"Specified controller type {controller_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
