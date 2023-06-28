import numpy as np
from test_utils import plot_trajs_c, plot_trajs_j

import matplotlib.pyplot as plt
import fancy_gym

import copy

from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name

init_c_pos = np.array([0.65, 0., 0.1645])
init_c_vel = np.array([0, 0, 0])
init_c_acc = np.array([0, 0, 0])
init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])
init_j_acc = np.array([0, 0, 0, 0, 0, 0, 0])


def dynamic_system_c(des_pos, des_vel, cur_pos, cur_vel) -> np.ndarray:
    return 100 * (des_pos - cur_pos) + 20 * (des_vel - cur_vel)


def dynamic_system_j(des_pos, des_vel, cur_pos, cur_vel) -> np.ndarray:
    pass


def forward():
    pass


def backward():
    pass


def optimize():
    pass


def test_env():
    env_kwargs = {'check_traj': False, 'check_step': False}

    # create env
    env = fancy_gym.make(env_id='7dof-hit', seed=0, **env_kwargs)
    env_info = env.env_info
    robot_model = copy.deepcopy(env_info['robot']['robot_model'])
    robot_data = copy.deepcopy(env_info['robot']['robot_data'])

    traj_c = np.load('traj_c.npy', allow_pickle=True)

    obs = env.reset()
    cur_j_pos = obs[6:13]
    cur_j_vel = obs[13:20]
    cur_j_acc = np.array([0, 0, 0])
    cur_c_pos = forward_kinematics(robot_model, robot_data, cur_j_pos)[0]
    jac = jacobian(robot_model, robot_data, cur_j_pos)[:3, :7]
    cur_c_vel = jac @ cur_j_vel
    print('aaa')
    for i, c in enumerate(traj_c):
        des_c_pos = c[0]
        des_c_vel = c[1]
        des_c_acc = dynamic_system_c(des_c_pos, des_c_vel, cur_c_pos, cur_c_vel)


if __name__ == "__main__":
    # traj_c = np.load('traj_c.npy', allow_pickle=True)
    # traj_j = np.load('traj_j.npy', allow_pickle=True)
    # plot_trajs_c(traj_c[:, 0], traj_c[:, 1])
    # plot_trajs_j(traj_j[:, 0], traj_j[:, 1], 0, 149, False, True, 7)

    test_env()

    # traj_c_cur = np.zeros([150, 3, 3])
    # cur_c_pos = init_c_pos
    # cur_c_vel = init_c_vel
    # cur_c_acc = init_c_acc
    # jerk = np.array([0, 0, 0])
    # for i, c in enumerate(traj_c):
    #     des_c_pos = c[0]
    #     des_c_vel = c[1]
    #     raw_c_acc = dynamic_system_c(des_c_pos, des_c_vel, cur_c_pos, cur_c_vel)
    #     des_c_acc = np.clip(raw_c_acc, cur_c_acc - 10, cur_c_acc + 10)
    #
    #     # jerk = np.clip((des_c_acc - cur_c_acc) / 0.02, -100, +100)
    #     # print(jerk)
    #     jerk = (des_c_acc - cur_c_acc) / 0.02
    #     cur_c_pos = cur_c_pos + cur_c_vel * 0.02 + 0.5 * cur_c_acc * 0.02**2 + 0.166 * jerk * 0.02**3
    #     cur_c_vel = cur_c_vel + cur_c_acc * 0.02 + 0.5 * jerk * 0.02**2
    #     cur_c_acc = cur_c_acc + jerk * 0.02
    #     traj_c_cur[i, 0] = cur_c_pos
    #     traj_c_cur[i, 1] = cur_c_vel
    #     traj_c_cur[i, 2] = cur_c_acc
    #
    # plot_trajs_j(traj_c_cur[:, 0], traj_c_cur[:, 1], 0, 149, False, False, 3)
    # plt.plot(traj_c[:, :, 0], traj_c[:, :, 1], color='r')
    # plt.plot(traj_c_cur[:, :2, 0], traj_c_cur[:, :2, 1], color='b')
    # plt.show()
    # print(traj_c.shape)

