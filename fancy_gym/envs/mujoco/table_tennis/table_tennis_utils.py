import numpy as np

jnt_pos_low = np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2])
jnt_pos_high = np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2])

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
    t_n = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 - 8 * (net_height - z) / g)) / 2.
    if x + v_x * t_n < 0.05:
        return False
    # check if ball landing position will violate x bounds
    t_l = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 + 8 * (z) / g)) / 2.
    if x + v_x * t_l > table_x_max:
        return False
    # check if ball landing position will violate y bounds
    if y + v_y * t_l > table_y_max or y + v_y * t_l < table_y_min:
        return False

    return True


def is_init_state_valid_batch(init_states):
    assert init_states.shape[1] == 3, "init_state must be a 6D vector (pos+vel),got {}".format(init_states)
    x = init_states[:, 0]
    y = init_states[:, 1]
    z = np.ones(init_states.shape[0]) * 1.75 - table_height + 0.1
    v_x = init_states[:, 2]
    v_y = np.ones(init_states.shape[0]) * 0
    v_z = np.ones(init_states.shape[0]) * 0.5

    cor_states = np.ones(init_states.shape[0])

    # check if the initial state is wrong
    x_wrong = np.where(x > -0.2)[0]
    if x_wrong.shape[0] != 0:
        cor_states[x_wrong] = False

    v_x_wrong = np.where(v_x < 0.)[0]
    if v_x_wrong.shape[0] != 0:
        cor_states[v_x_wrong] = False

    # check if the ball can pass the net
    t_n = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 - 8 * (net_height - z) / g)) / 2.
    can_not_pass_net = np.where(x + v_x * t_n < 0.05)[0]
    if can_not_pass_net.shape[0] != 0:
        cor_states[can_not_pass_net] = 0

    # check if ball landing position will violate x bounds
    t_l = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 + 8 * (z) / g)) / 2.
    x_bounds_wrong = np.where(x + v_x * t_l > table_x_max)[0]
    if x_bounds_wrong.shape[0] != 0:
        cor_states[x_bounds_wrong] = 0

    # check if ball landing position will violate y bounds
    ball_landing_pos_wrong_1 = np.where(y + v_y * t_l > table_y_max)[0]
    ball_landing_pos_wrong_2 = np.where(y + v_y * t_l < table_y_min)[0]
    if ball_landing_pos_wrong_1.shape[0] != 0:
        cor_states[ball_landing_pos_wrong_1] = 0
    if ball_landing_pos_wrong_2.shape[0] != 0:
        cor_states[ball_landing_pos_wrong_2] = 0
    return cor_states


def is_init_state_valid_only_rndm_pos_batch(init_states):
    assert init_states.shape[1] == 2, "init_state must be a 6D vector (pos+vel),got {}".format(init_states)
    x = init_states[:, 0]
    y = init_states[:, 1]
    z = 1.75 - table_height + 0.1
    v_x = 2.5
    v_y = 0.
    v_z = 0.5

    cor_states = np.ones(init_states.shape[0])

    # check if the initial state is wrong
    x_wrong = np.where(x > -0.2)[0]
    if x_wrong.shape[0] != 0:
        cor_states[x_wrong] = False
        return cor_states

    # check if the ball can pass the net
    t_n = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 - 8 * (net_height - z) / g)) / 2.
    can_not_pass_net = np.where(x + v_x * t_n < 0.05)[0]
    if can_not_pass_net.shape[0] != 0:
        cor_states[can_not_pass_net] = 0

    # check if ball landing position will violate x bounds
    t_l = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 + 8 * (z) / g)) / 2.
    x_bounds_wrong = np.where(x + v_x * t_l > table_x_max)[0]
    if x_bounds_wrong.shape[0] != 0:
        cor_states[x_bounds_wrong] = 0

    # check if ball landing position will violate y bounds
    ball_landing_pos_wrong_1 = np.where(y + v_y * t_l > table_y_max)[0]
    ball_landing_pos_wrong_2 = np.where(y + v_y * t_l < table_y_min)[0]
    if ball_landing_pos_wrong_1.shape[0] != 0:
        cor_states[ball_landing_pos_wrong_1] = 0
    if ball_landing_pos_wrong_2.shape[0] != 0:
        cor_states[ball_landing_pos_wrong_2] = 0
    return cor_states


def is_init_state_valid_only_rndm_pos(init_state):
    assert len(init_state) == 2, "init_state must be a 6D vector (pos+vel),got {}".format(init_state)
    x = init_state[0]
    y = init_state[1]
    z = 1.75 - table_height + 0.1
    v_x = 2.5
    v_y = 0.
    v_z = 0.5

    # check if the initial state is wrong
    if x > -0.2:
        return 0
    # check if the ball can pass the net
    t_n = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 - 8 * (net_height - z) / g)) / 2.
    if x + v_x * t_n < 0.05:
        return 0
    # check if ball landing position will violate x bounds
    t_l = (-2. * (-v_z) / g + np.sqrt(4 * (v_z ** 2) / g ** 2 + 8 * (z) / g)) / 2.
    if x + v_x * t_l > table_x_max:
        return 0
    # check if ball landing position will violate y bounds
    if y + v_y * t_l > table_y_max or y + v_y * t_l < table_y_min:
        return 0

    return 1


def check_init_states_valid_only_rndm_pos_function(init_states):
    cor_states = is_init_state_valid_only_rndm_pos_batch(init_states)
    ref_cor_states = np.zeros(init_states.shape[0])
    for i in range(init_states.shape[0]):
        ref_cor_states[i] = is_init_state_valid_only_rndm_pos(init_states[i])
    return cor_states - ref_cor_states


def check_init_states_valid_function(init_states):
    cor_states = is_init_state_valid_batch(init_states)
    ref_cor_states = np.zeros(init_states.shape[0])
    for i in range(init_states.shape[0]):
        ref_cor_states[i] = is_init_state_valid(init_states[i])
    return cor_states - ref_cor_states


def magnus_force(top_spin=0.0, side_spin=0.0, v_ball=np.zeros(3), v_wind=np.zeros(3)):
    rho = 1.225  # Air density
    A = 1.256 * 10e-3  # Cross-section area of ball
    C_l = 4.68 * 10e-4 - 2.0984 * 10e-5 * (np.linalg.norm(v_ball) - 50)  # Lift force coeffient or simply 1.23
    w = np.array([0.0, top_spin, side_spin])  # Angular velocity of ball
    f_m = 0.5 * rho * A * C_l * np.linalg.norm(v_ball - v_wind) * np.cross(w, v_ball - v_wind)
    return f_m
