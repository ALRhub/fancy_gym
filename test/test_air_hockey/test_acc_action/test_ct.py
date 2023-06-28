import numpy as np
from test_utils import plot_trajs_c, plot_trajs_j

import matplotlib.pyplot as plt
import fancy_gym

import copy

from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name

import scipy.linalg
from scipy import sparse
import osqp


init_c_pos = np.array([0.65, 0., 0.1645])
init_c_vel = np.array([0, 0, 0])
init_c_acc = np.array([0, 0, 0])
init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])
init_j_acc = np.array([0, 0, 0, 0, 0, 0, 0])


def optimize_step(x_des, v_des, q_cur, dq_cur, robot_model, robot_data, env_info):
    anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])
    dq_anchor = np.zeros(7)

    x_cur = forward_kinematics(robot_model, robot_data, q_cur)[0]
    jac = jacobian(robot_model, robot_data, q_cur)[:3, :7]
    N_J = scipy.linalg.null_space(jac)
    b = np.linalg.lstsq(jac, v_des + ((x_des - x_cur) / 0.02), rcond=None)[0]

    P = (N_J.T @ np.diag(anchor_weights) @ N_J) / 2
    q = (b - dq_anchor).T @ np.diag(anchor_weights) @ N_J
    A = N_J.copy()
    u = np.minimum(env_info['robot']['joint_vel_limit'][1] * 0.92,
                   (env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / env_info['dt']) - b
    l = np.maximum(env_info['robot']['joint_vel_limit'][0] * 0.92,
                   (env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / env_info['dt']) - b

    if np.any(u <= l):
        u = env_info['robot']['joint_vel_limit'][1] * 0.92 - b
        l = env_info['robot']['joint_vel_limit'][0] * 0.92 - b

    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

    result = solver.solve()
    if result.info.status == 'solved':
        return True, N_J @ result.x + b
    else:
        vel_u = env_info['robot']['joint_vel_limit'][1] * 0.92
        vel_l = env_info['robot']['joint_vel_limit'][0] * 0.92
        # return False, np.clip(b, vel_l, vel_u)
        return False, b


def dynamic_system(des_j_pos, des_j_vel, cur_j_pos, cur_j_vel):
    return 200 * (des_j_pos - cur_j_pos) + 40 * (des_j_vel - cur_j_vel)


def test_env():
    env_kwargs = {'check_traj': False, 'check_step': False}

    # create env
    env = fancy_gym.make(env_id='7dof-hit', seed=0, **env_kwargs)
    env_info = env.env_info
    robot_model = copy.deepcopy(env_info['robot']['robot_model'])
    robot_data = copy.deepcopy(env_info['robot']['robot_data'])

    # reference traj_c
    traj_c = np.load('traj_c.npy', allow_pickle=True)

    obs = env.reset()
    env.render(mode='human')
    cur_j_pos = obs[6:13]
    cur_j_vel = obs[13:20]
    cur_j_acc = np.zeros(7)
    for i, c in enumerate(traj_c):
        des_c_pos = c[0]
        des_c_vel = c[1]
        success, des_j_vel = optimize_step(des_c_pos, des_c_vel, cur_j_pos, cur_j_vel,
                                           robot_model, robot_data, env_info)
        des_j_pos = cur_j_pos + (cur_j_vel + des_j_vel) / 2 * 0.02
        des_j_acc = dynamic_system(des_j_pos, des_j_vel, cur_j_pos, cur_j_vel)

        # jerk = np.clip((des_c_acc - cur_c_acc) / 0.02, -100, +100)
        jerk = (des_j_acc - cur_j_acc) / 0.02
        des_j_pos = cur_j_pos + cur_j_vel * 0.02 + 0.5 * cur_j_acc * 0.02**2 + 0.166 * jerk * 0.02**3
        des_j_vel = cur_j_vel + cur_j_acc * 0.02 + 0.5 * jerk * 0.02**2
        des_j_acc = cur_j_acc + jerk * 0.02

        act = np.vstack([des_j_pos, des_j_vel])
        obs, rew, done, info = env.step(act)
        env.render(mode='human')

        cur_j_pos = obs[6:13]
        cur_j_vel = obs[13:20]
        cur_j_acc = des_j_acc


if __name__ == '__main__':
    test_env()
