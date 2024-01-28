import numpy as np

jnt_pos_low = np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2])
jnt_pos_high = np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2])

jnt_vel_low = np.ones(7) * -7
jnt_vel_high = np.ones(7) * 7

delay_bound = [0.05, 0.15]
tau_bound = [0.5, 1.5]

net_height = 0.1
table_height = 0.77
table_x_min = -1.1
table_x_max = 1.1
table_y_min = -0.6
table_y_max = 0.6
g = 9.81

def is_init_state_valid(init_state):
    assert len(init_state) == 6, "init_state must be a 6D vector (pos+vel),got {}".format(init_state)
    x = init_state[0]
    y = init_state[1]
    z = init_state[2] - table_height + 0.1
    v_x = init_state[3]
    v_y = init_state[4]
    v_z = init_state[5]

    # check if the initial state is wrong
    if x > -0.2:
        return False
    # check if the ball velocity direction is wrong
    if v_x < 0.:
        return False
    # check if the ball can pass the net
    t_n = (-2.*(-v_z)/g + np.sqrt(4*(v_z**2)/g**2 - 8*(net_height-z)/g))/2.
    if x + v_x * t_n < 0.05:
        return False
    # check if ball landing position will violate x bounds
    t_l = (-2.*(-v_z)/g + np.sqrt(4*(v_z**2)/g**2 + 8*(z)/g))/2.
    if x + v_x * t_l > table_x_max:
        return False
    # check if ball landing position will violate y bounds
    if y + v_y * t_l > table_y_max or y + v_y * t_l < table_y_min:
        return False

    return True

def magnus_force(top_spin=0.0, side_spin=0.0, v_ball=np.zeros(3), v_wind=np.zeros(3)):
    rho = 1.225        # Air density
    A = 1.256 * 10e-3  # Cross-section area of ball
    C_l = 4.68 * 10e-4 - 2.0984 * 10e-5 * (np.linalg.norm(v_ball) - 50)  # Lift force coeffient or simply 1.23
    w = np.array([0.0, top_spin, side_spin]) # Angular velocity of ball
    f_m = 0.5 * rho * A * C_l * np.linalg.norm(v_ball-v_wind) * np.cross(w, v_ball-v_wind)
    return f_m