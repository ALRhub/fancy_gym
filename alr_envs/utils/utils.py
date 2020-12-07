import numpy as np


def angle_normalize(x, type="deg"):
    """
    normalize angle x to [-pi,pi].
    Args:
        x: Angle in either degrees or radians
        type: one of "deg" or "rad" for x being in degrees or radians

    Returns:

    """
    if type == "deg":
        return ((x + np.pi) % (2 * np.pi)) - np.pi
    elif type == "rad":
        two_pi = 2 * np.pi
        return x - two_pi * np.floor((x + np.pi) / two_pi)
    else:
        raise ValueError(f"Invalid type {type}. Choose on of 'deg' or 'rad'.")
