import time
import fancy_gym
import torch
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from test_utils import plot_trajs, plot_trajs_cart
from test_traj_generator import TrajectoryGenerator
from test_traj_optimizer import TrajectoryOptimizer

# phase_generator_kwargs = {'phase_generator_type': 'linear',
#                           'tau': 3.0}
# basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
#                           'num_basis': 5,
#                           'num_basis_zero_start': 3,
#                           'num_basis_zero_goal': 0,
#                           'basis_bandwidth_factor': 3.0}
# trajectory_generator_kwargs = {'trajectory_generator_type': 'promp',
#                                'action_dim': 2,
#                                'weights_scale': 1}
phase_generator_kwargs = {'phase_generator_type': 'exp',
                          'delay': 0,
                          'tau': 3.0,
                          'alpha_phase': 3}
basis_generator_kwargs = {'basis_generator_type': 'prodmp',
                          'num_basis': 4,
                          'alpha': 15,
                          'basis_bandwidth_factor': 3.0}
trajectory_generator_kwargs = {'trajectory_generator_type': 'prodmp',
                               'action_dim': 2,
                               'weights_scale': 1.0,
                               'goal_scale': 1.0,
                               'disable_weights': False,
                               'disable_goal': False,
                               'relative_goal': False,
                               'auto_scale_basis': True}


def test_cart_agent(env_id='7dof-hit', seed=0):
    env_kwargs = {'check_traj': False, 'check_step': False}

    # create env
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
    env_info = env.env_info

    # init condition
    init_c_pos = np.array([0.65, 0., 0.1645])
    init_c_vel = np.array([0, 0, 0])
    init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])

    traj_gen = TrajectoryGenerator(env_info)
    traj_opt = TrajectoryOptimizer(env_info)

    plt.vlines(-env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)
    plt.vlines(+env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)
    plt.hlines(-env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    plt.hlines(+env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    for _ in range(1):
        weights = 0.5 * (2 * np.random.rand(20) - 1)
        # weights = np.array([0.4758, 0.7876, 0.4221, 0.5428, 0.4740,
        #                     0.3129, 0.0619, 0.4272, 0.0162, 0.0826])
        pos, vel = traj_gen.generate_trajectory(weights, init_c_pos, init_c_vel)
        plt.plot(pos[:, 0] - 1.51, pos[:, 1], color='red')
    plt.show()

    # for _ in range(1):
    #     weights = 0.5 * (2 * np.random.rand(20) - 1)
    #     weights = np.array([0.4758, 0.7876, 0.4221, 0.5428, 0.4740,
    #                         0.3129, 0.0619, 0.4272, 0.0162, 0.0826])
    #     pos, vel = traj_gen.generate_trajectory(weights, init_c_pos, init_c_vel)
    #     traj_c = np.hstack([pos, vel])
    #
    #     success, j_pos_traj = traj_opt.optimize_trajectory(traj_c, init_j_pos, init_j_vel, None)
    #     t = np.linspace(0, j_pos_traj.shape[0], j_pos_traj.shape[0] + 1) * 0.02
    #     f = CubicSpline(t, np.vstack([init_j_pos, j_pos_traj]), axis=0, bc_type=((1, init_j_vel),
    #                                                                             (2, np.zeros_like(init_j_vel))))
    #     df = f.derivative(1)
    #     traj_j = np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
    #
    #     c_pos = np.zeros([traj_j.shape[0], 3])
    #     for i, j in enumerate(traj_j):
    #         c_pos[i] = forward_kinematics(env_info['robot']['robot_model'], env_info['robot']['robot_data'], j[0])[0]
    #
    #     plot_trajs(np.vstack([init_j_pos, traj_j[:, 0]]), np.vstack([init_j_vel, traj_j[:, 1]]), 0, 150, False, True, dof=7)
    #     #
    #     plt.vlines(-env_info['table']['length']/2, ymin=-0.6, ymax=+0.6)
    #     plt.vlines(+env_info['table']['length']/2, ymin=-0.6, ymax=+0.6)
    #     plt.hlines(-env_info['table']['width']/2, xmin=-1.1, xmax=+1.1)
    #     plt.hlines(+env_info['table']['width']/2, xmin=-1.1, xmax=+1.1)
    #     plt.plot(pos[:, 0] - 1.51, pos[:, 1], color='red')
    #     plt.plot(c_pos[:, 0] - 1.51, c_pos[:, 1])
    #     # t = np.linspace(0.02, 3, pos.shape[0])
    #     # plt.plot(t, pos[:, 0])
    #     plt.show()
    #     print(weights)
    #
    #     rews = []
    #     jerks = []
    #     constrs = {'j_pos': [], 'j_vel': [], 'ee': []}
    #     obs = env.reset()
    #     cnt = 0
    #     for j in traj_j:
    #         act = np.hstack([j[0], j[1]])
    #         obs_, rew, done, info = env.step(act)
    #         env.render(mode="human")
    #         cnt += 1
    #
    #         rews.append(rew)
    #         jerks.append(info['jerk_violation'])
    #         constrs['j_pos'].append(info['j_pos_violation'])
    #         constrs['j_vel'].append(info['j_vel_violation'])
    #         constrs['ee'].append(info['ee_violation'])
    #
    #         if done:
    #             print('Return: ', np.sum(rews))
    #             print('Jerks: ', np.sum(jerks))
    #             print('constr_j_pos: ', np.sum(constrs['j_pos']))
    #             print('constr_j_vel: ', np.sum(constrs['j_vel']))
    #             print('constr_ee: ', np.sum(constrs['ee']))
    #             print(cnt)
    #             break


if __name__ == "__main__":
    # test_traj_generator()
    test_traj_optimizer()
    # test_cart_agent()
