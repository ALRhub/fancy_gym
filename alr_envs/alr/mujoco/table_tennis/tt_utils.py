import numpy as np


def ball_init(random=False, context_range=None):
    if random:
        dx = np.random.uniform(-0.1, 0.1)           # TODO: clarify these numbers?
        dy = np.random.uniform(-0.1, 0.1)           # TODO: clarify these numbers?
        dz = np.random.uniform(-0.1, 0.1)           # TODO: clarify these numbers?

        v_x = np.random.uniform(1.7, 1.8)
        v_y = np.random.uniform(0.7, 0.8)
        v_z = np.random.uniform(0.1, 0.2)
    else:
        dx = 1
        dy = 0
        dz = 0.05

        v_x = 2.5
        v_y = 2
        v_z = 0.5

    initial_x = 0 + dx - 1.2
    initial_y = -0.2 + dy - 0.6
    initial_z = 0.3 + dz + 1.5
    initial_ball_state = np.array([initial_x, initial_y, initial_z, v_x, v_y, v_z])
    return initial_ball_state
