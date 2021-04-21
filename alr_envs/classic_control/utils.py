def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0]) > 1e-12


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def check_self_collision(line_points):
    for i, line1 in enumerate(line_points):
        for line2 in line_points[i + 2:, :, :]:
            # if line1 != line2:
            if intersect(line1[0], line1[-1], line2[0], line2[-1]):
                return True
    return False

