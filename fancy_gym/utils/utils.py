from collections.abc import Mapping, MutableMapping

import numpy as np
import torch as ch


def angle_normalize(x, type="deg"):
    """
    normalize angle x to [-pi,pi].
    Args:
        x: Angle in either degrees or radians
        type: one of "deg" or "rad" for x being in degrees or radians

    Returns:

    """

    if type not in ["deg", "rad"]: raise ValueError(f"Invalid type {type}. Choose one of 'deg' or 'rad'.")

    if type == "deg":
        x = np.deg2rad(x)  # x * pi / 180

    two_pi = 2 * np.pi
    return x - two_pi * np.floor((x + np.pi) / two_pi)


def get_numpy(x: ch.Tensor):
    """
    Returns numpy array from torch tensor
    Args:
        x:

    Returns:

    """
    return x.detach().cpu().numpy()


def nested_update(base: MutableMapping, update):
    """
    Updated method for nested Mappings
    Args:
        base: main Mapping to be updated
        update: updated values for base Mapping

    """
    for k, v in update.items():
        base[k] = nested_update(base.get(k, {}), v) if isinstance(v, Mapping) else v
    return base

