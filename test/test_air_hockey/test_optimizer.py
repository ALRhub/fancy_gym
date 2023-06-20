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
    def __init__(self, env_info):
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        if self.n_joints == 3:
            self.anchor_weights = np.ones(3)
        else:
            self.anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])
            # self.anchor_weights = np.array([10., 1., 10., 1., 10., 1., 1.])

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        joint_trajectory = np.tile(np.concatenate([q_start]), (cart_traj.shape[0], 1))
        if len(cart_traj) > 0:
            q_cur = q_start.copy()
            dq_cur = dq_start.copy()

            for i, des_point in enumerate(cart_traj):
                if q_anchor is None:
                    dq_anchor = 0
                else:
                    dq_anchor = (q_anchor - q_cur)

                success, dq_next = self._solve_aqp(des_point[:3], des_point[3:6], q_cur, dq_anchor)
                print(success)
                if not success:
                    # return success, []
                    q_cur += (dq_cur + dq_next) / 2 * self.env_info['dt']
                    # q_cur += dq_next * self.env_info['dt']
                    dq_cur = dq_next
                    joint_trajectory[i] = q_cur.copy()
                else:
                    q_cur += (dq_cur + dq_next) / 2 * self.env_info['dt']
                    # q_cur += dq_next * self.env_info['dt']
                    dq_cur = dq_next
                    joint_trajectory[i] = q_cur.copy()
            return True, joint_trajectory
        else:
            return False, []

    def _solve_aqp(self, x_des, v_des, q_cur, dq_anchor):
        x_cur = forward_kinematics(self.robot_model, self.robot_data, q_cur)[0]
        jac = jacobian(self.robot_model, self.robot_data, q_cur)[:3, :self.n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, v_des + ((x_des - x_cur) / self.env_info['dt']), rcond=None)[0]

        P = (N_J.T @ np.diag(self.anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(self.anchor_weights) @ N_J
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


class JointTrajectoryOptimizer:
    def __int__(self, env_info):
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        if self.n_joints == 3:
            self.weights = np.ones(3)
        else:
            self.weights = np.array([10., 1., 10., 1., 10., 1., 10.])

        self.q_start = np.zeros(self.n_joints)
        self.dq_start = np.zeros(self.n_joints)

    def optimize_joint_trajectory(self, q_traj, dq_traj, q_start=None, dq_start=None):
        q_start = self.q_start if q_start is None else q_start
        dq_start = self.dq_start if dq_start is None else dq_start
        q_traj_opt = np.tile(np.concatenate([q_start]), (q_traj.shape[0], 1))
        dq_traj_opt = np.tile(np.concatenate([dq_start]), (dq_traj.shape[0], 1))

        q_cur = q_start
        dq_cur = dq_start
        for i, q_des, dq_des in enumerate(zip(q_traj, dq_traj)):
            success, q_next, dq_next = self._solve_qp(q_des, dq_des, q_cur, dq_cur)
            if success:
                q_traj_opt[i] = q_next
                dq_traj_opt[i] = dq_next

    def _solve_qp(self, q_des, dq_des, q_cur, dq_cur):
        jac = jacobian(self.robot_model, self.robot_data, q_cur)
        # todo

        q_opt = q_des
        dq_opt = dq_des
        return True, q_opt, dq_opt


if __name__ == '__main__':
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardV2',
                  'check_traj': False, 'check_step': False}
    env = fancy_gym.make(env_id='7dof-hit', seed=0, **env_kwargs)
    env_info = env.env_info

