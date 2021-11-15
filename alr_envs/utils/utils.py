import numpy as np


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
