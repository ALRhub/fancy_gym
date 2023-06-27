import time
import copy
import osqp
import fancy_gym
import numpy as np
import torch
import scipy.linalg
from scipy import sparse
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name


class TrajectoryOptimizer:
    def __init__(self, env_info, optimizer_kwargs=None):
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        self.dt = env_info['dt']

        self.optimizer_kwargs = optimizer_kwargs

        if self.n_joints == 3:
            self.weights = np.ones(3)
        else:
            self.weights = np.array([10., 1., 10., 1., 10., 10., 1.])

        self.init_j_acc = np.zeros(self.n_joints)

    def reset(self):
        self.init_j_acc = np.zeros(self.n_joints)

    def optimize_trajectory(self, traj_c_pos, traj_c_vel, init_j_pos, init_j_vel):
        traj_j_pos = np.tile(np.concatenate([init_j_pos]), (traj_c_pos.shape[0], 1))
        traj_j_vel = np.tile(np.concatenate([init_j_vel]), (traj_c_vel.shape[0], 1))

        cur_j_pos = init_j_pos.copy()
        cur_j_vel = init_j_vel.copy()

        for i, [des_c_pos, des_c_vel] in enumerate(zip(traj_c_pos, traj_c_vel)):
            success, next_j_vel = self._solve_qp(des_c_pos, des_c_vel, cur_j_pos, cur_j_vel)

            cur_j_pos = (cur_j_vel + next_j_vel) / 2 * self.dt
            cur_j_vel = next_j_vel

            traj_j_pos[i] = cur_j_pos.copy()
            traj_j_vel[i] = cur_j_vel.copy()

        # t = np.linspace(0, traj_j_pos.shape[0], traj_j_pos.shape[0] + 1) * self.dt
        # f = CubicSpline(t, np.vstack([cur_j_pos, traj_j_pos]), axis=0, bc_type=((1, init_j_vel),
        #                                                                         (2, self.init_j_acc)))
        # df = f.derivative(1)
        # ddf = f.derivative(2)

        return True, traj_j_pos, traj_j_vel

    def _solve_qp(self, des_c_pos, des_c_vel, cur_j_pos, cur_j_vel):
        cur_c_pos = forward_kinematics(self.robot_model, self.robot_data, cur_j_pos)[0]
        jac = jacobian(self.robot_model, self.robot_data, cur_j_pos)[:3, :self.n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, des_c_vel + ((des_c_pos - cur_c_pos) / self.dt), rcond=None)[0]

        P = (N_J.T @ np.diag(self.weights) @ N_J) / 2
        q = b.T @ np.diag(self.weights) @ N_J
        A = N_J.copy()
        # u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.92,
        #                (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        # l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.92,
        #                (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        u = self.env_info['robot']['joint_vel_limit'][1] * 0.92 - b
        l = self.env_info['robot']['joint_vel_limit'][0] * 0.92 - b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            vel_u = self.env_info['robot']['joint_vel_limit'][1] * 0.92
            vel_l = self.env_info['robot']['joint_vel_limit'][0] * 0.92
            # return False, np.clip(b, vel_l, vel_u)
            return False, b

# def test_traj_optimizer(env_id='7dof-hit', seed=0):
#     env_kwargs = {'check_traj': False, 'check_step': False}
#
#     # create env
#     env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
#     env_info = env.env_info
#
#     # init condition
#     init_c_pos = np.array([0.65, 0., 0.1645])
#     init_c_vel = np.array([0, 0, 0])
#     init_c_acc = np.array([0, 0, 0])
#     init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
#     init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])
#     init_j_acc = np.array([0, 0, 0, 0, 0, 0, 0])
#
#     traj_gen = TrajectoryGenerator(env_info)
#     traj_opt = TrajectoryOptimizer(env_info)
#
#     init_time = [0, 0.5, 1.0, 1.5]
#     traj_length = [25, 25, 25, 75]
#     colors = ['red', 'blue', 'black', 'green', 'yellow']
#     cur_c_pos = init_c_pos
#     cur_c_vel = init_c_vel
#     cur_c_acc = init_c_acc
#     cur_j_pos = init_j_pos
#     cur_j_vel = init_j_vel
#     cur_j_acc = init_j_acc
#     list_c_pos = []
#     list_c_vel = []
#     list_j_pos = []
#     list_j_vel = []
#     obs = env.reset()
#     for i in range(4):
#         init_t = init_time[i]
#         traj_l = traj_length[i]
#
#         weights = 0.1 * (2 * np.random.rand(10) - 1)
#         weights[4] = (np.random.rand(1) + 0.5)
#         weights[9] = (0.5 * (2 * np.random.rand(1) - 1))
#         c_pos, c_vel = traj_gen.generate_trajectory(weights, init_t, cur_c_pos, cur_c_vel)
#         list_c_pos.append(c_pos[:traj_l])
#         list_c_vel.append(c_vel[:traj_l])
#         cur_c_pos = c_pos[traj_l - 1].copy()
#         cur_c_vel = c_vel[traj_l - 1].copy()
#         success, j_pos = traj_opt.optimize_trajectory(np.hstack([c_pos[:traj_l], c_vel[:traj_l]]),
#                                                       cur_j_pos, cur_j_vel, None)
#         t = np.linspace(0, j_pos.shape[0], j_pos.shape[0] + 1) * 0.02
#         f = CubicSpline(t, np.vstack([cur_j_pos, j_pos]), axis=0, bc_type=((1, cur_j_vel), (2, cur_j_acc)))
#         df = f.derivative(1)
#         ddf = f.derivative(2)
#
#         j_pos = f(t[1:])
#         j_vel = df(t[1:])
#         j_acc = ddf(t[1:])
#
#         cur_j_pos = j_pos[-1]
#         cur_j_vel = j_vel[-1]
#         cur_j_acc = np.zeros(7)
#
#         list_j_pos.append(j_pos)
#         list_j_vel.append(j_vel)
#
#         for p, v in zip(j_pos, j_vel):
#             act = np.hstack([p, v])
#             obs_, rew, done, info = env.step(act)
#             env.render(mode="human")
#
#     # c_pos = np.concatenate(list_c_pos)
#     # c_vel = np.concatenate(list_c_vel)
#     # success, j_pos = traj_opt.optimize_trajectory(np.hstack([c_pos, c_vel]),
#     #                                               cur_j_pos, cur_j_vel, None)
#     # t = np.linspace(0, j_pos.shape[0], j_pos.shape[0] + 1) * 0.02
#     # f = CubicSpline(t, np.vstack([cur_j_pos, j_pos]), axis=0, bc_type=((1, cur_j_vel), (2, cur_j_acc)))
#     # df = f.derivative(1)
#     # ddf = f.derivative(2)
#     #
#     # j_pos = f(t[1:])
#     # j_vel = df(t[1:])
#     # j_acc = ddf(t[1:])
#     #
#     # list_j_pos.append(j_pos)
#     # list_j_vel.append(j_vel)
#     # #
#     # for p, v in zip(j_pos, j_vel):
#     #     act = np.hstack([p, v])
#     #     obs_, rew, done, info = env.step(act)
#     #     env.render(mode="human")
#
#     plot_trajs(np.concatenate(list_j_pos, axis=0), np.concatenate(list_j_vel, axis=0),
#                plot_constrs=True, plot_sampling=False, dof=7)


if __name__ == '__main__':
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardV2',
                  'check_traj': False, 'check_step': False}
    env = fancy_gym.make(env_id='7dof-hit', seed=0, **env_kwargs)
    env_info = env.env_info

