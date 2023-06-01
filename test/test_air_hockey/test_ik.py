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

from test_envs import plot_trajs

from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name
from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator

phase_generator_kwargs = {'phase_generator_type': 'linear'}
basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
                          'num_basis': 5,
                          'num_basis_zero_start': 3,
                          'num_basis_zero_goal': 0,
                          'basis_bandwidth_factor': 3.0}
trajectory_generator_kwargs = {'trajectory_generator_type': 'promp',
                               'action_dim': 2,
                               'weights_scale': 1}


def get_numpy(x: torch.Tensor):
    """
    Returns numpy array from torch tensor
    Args:
        x:

    Returns:

    """
    return x.detach().cpu().numpy()


class TrajectoryGenerator:
    def __init__(self, env__info):
        self.env_info = env__info

        self.init_time = np.array(0)
        self.init_pos = np.array([6.49932054e-01, 2.05280684e-05, 1.00000000e-01])
        self.init_vel = np.array([0, 0, 0])

        self.dt = 0.001
        self.duration = 3.0

        phase_gen = get_phase_generator(**phase_generator_kwargs)
        basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_generator_kwargs)
        self.traj_gen = get_trajectory_generator(basis_generator=basis_gen, **trajectory_generator_kwargs)

    def generate_trajectory(self, weights):
        self.traj_gen.reset()
        self.traj_gen.set_params(weights)
        self.traj_gen.set_initial_conditions(self.init_time, self.init_pos[:2], self.init_vel[:2])
        self.traj_gen.set_duration(self.duration, self.dt)

        # get position and velocity
        position = get_numpy(self.traj_gen.get_traj_pos())
        velocity = get_numpy(self.traj_gen.get_traj_vel())

        if self.dt == 0.001:
            position = position[19::20].copy()
            velocity = velocity[19::20].copy()

        return position, velocity


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
        b = np.linalg.lstsq(jac, ((x_des - x_cur) / self.env_info['dt']), rcond=None)[0]

        P = (N_J.T @ np.diag(self.anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(self.anchor_weights) @ N_J
        A = N_J.copy()
        u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.92,
                       (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.92,
                       (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            vel_u = self.env_info['robot']['joint_vel_limit'][1] * 0.92
            vel_l = self.env_info['robot']['joint_vel_limit'][0] * 0.92
            return False, np.clip(b, vel_l, vel_u)


def test_cart_agent(env_id='3dof-hit', seed=0):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    traj_gen = TrajectoryGenerator(env_info)
    traj_opt = TrajectoryOptimizer(env_info)

    for _ in range(10):
        # weights = np.ones(10) * 0.1
        weights = np.random.random(10) * 0.5
        pos, vel = traj_gen.generate_trajectory(weights)

        pos = np.hstack([pos, 1e-1 * np.ones([pos.shape[0], 1])])
        vel = np.hstack([vel, np.zeros([vel.shape[0], 1])])
        c_traj = np.hstack([pos, vel])

        q_start = np.array([-1.1557072, 1.300244, 1.4428041])
        q_anchor = np.array([-1.15570723, 1.30024401, 1.44280414])
        dq_start = np.array([0, 0, 0])

        actions = []
        for i in range(1):
            s = 150 * (i + 0)
            e = 150 * (i + 1)
            success, j_pos_traj = traj_opt.optimize_trajectory(c_traj[s:e], q_start, dq_start, None)
            t = np.linspace(0, j_pos_traj.shape[0], j_pos_traj.shape[0] + 1) * 0.02
            f = CubicSpline(t, np.vstack([q_start, j_pos_traj]), axis=0, bc_type=((1, dq_start),
                                                                                  (2, np.zeros_like(dq_start))))
            df = f.derivative(1)
            j_traj = np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
            q_start = j_traj[0, 0]
            dq_start = j_traj[0, 1]
            actions.append(j_traj)
            # print(success)

        actions = np.vstack(actions)
        c_pos = np.zeros([actions.shape[0], 3])
        for j, a in enumerate(actions):
            c_pos[j] = forward_kinematics(env_info['robot']['robot_model'], env_info['robot']['robot_data'], a[0])[0]

        plot_trajs(actions[:, 0], actions[:, 1], 0, 149, False, True)

        plt.vlines(-env_info['table']['length']/2, ymin=-0.6, ymax=+0.6)
        plt.vlines(+env_info['table']['length']/2, ymin=-0.6, ymax=+0.6)
        plt.hlines(-env_info['table']['width']/2, xmin=-1.1, xmax=+1.1)
        plt.hlines(+env_info['table']['width']/2, xmin=-1.1, xmax=+1.1)
        plt.plot(pos[:, 0] - 1.51, pos[:, 1], color='red')
        plt.plot(c_pos[:, 0] - 1.51, c_pos[:, 1])
        plt.show()
        print(weights)


if __name__ == "__main__":
    test_cart_agent()