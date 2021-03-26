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
        x = np.deg2rad(x) # x * pi / 180

    two_pi = 2 * np.pi
    return x - two_pi * np.floor((x + np.pi) / two_pi)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0]) > 1e-12


def intersect(A, B, C, D):
    """
    Return true if line segments AB and CD intersects
    Args:
        A: start point line one
        B: end point line one
        C: start point line two
        D: end point line two

    Returns:

    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def check_self_collision(line_points):
    for i, line1 in enumerate(line_points):
        for line2 in line_points[i + 2:, :, :]:
            # if line1 != line2:
            if intersect(line1[0], line1[-1], line2[0], line2[-1]):
                return True
    return False
