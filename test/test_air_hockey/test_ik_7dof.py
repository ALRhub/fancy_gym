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
                          'alpha': 25,
                          'basis_bandwidth_factor': 3.0}
trajectory_generator_kwargs = {'trajectory_generator_type': 'prodmp',
                               'action_dim': 2,
                               'weights_scale': 1.0,
                               'goal_scale': 1.0,
                               'disable_weights': False,
                               'disable_goal': False,
                               'relative_goal': False,
                               'auto_scale_basis': True}


def get_numpy(x: torch.Tensor):
    """
    Returns numpy array from torch tensor
    Args:
        x:

    Returns:

    """
    return x.detach().cpu().numpy()


class TrajectoryGenerator:
    def __init__(self, env_info):
        self.env_info = env_info
        if '3dof' in env_info['env_name']:
            self.n_jnts = 3
            self.init_pos = np.array([0.65, 0.0, 0.1])
        else:
            self.n_jnts = 7
            self.init_pos = np.array([0.65, 0.0, 0.1645])

        self.init_vel = np.array([0, 0, 0])
        self.init_time = np.array(0)

        self.dt = 0.001
        self.duration = 4.0

        phase_gen = get_phase_generator(**phase_generator_kwargs)
        basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_generator_kwargs)
        self.traj_gen = get_trajectory_generator(basis_generator=basis_gen, **trajectory_generator_kwargs)

    def generate_trajectory(self, weights, init_time=None, init_pos=None, init_vel=None):
        init_time = self.init_time if init_time is None else init_time
        init_pos = self.init_pos if init_pos is None else init_pos
        init_vel = self.init_vel if init_vel is None else init_vel

        self.traj_gen.reset()
        self.traj_gen.set_params(weights)
        self.traj_gen.set_initial_conditions(init_time, init_pos[:2], init_vel[:2])
        self.traj_gen.set_duration(self.duration, self.dt)

        # get position and velocity
        position = get_numpy(self.traj_gen.get_traj_pos())
        velocity = get_numpy(self.traj_gen.get_traj_vel())

        if self.dt == 0.001:
            position = position[19::20].copy()
            velocity = velocity[19::20].copy()

        if self.n_jnts == 3:
            position = np.hstack([position, 1e-1 * np.ones([position.shape[0], 1])])
        else:
            position = np.hstack([position, 0.1645 * np.ones([position.shape[0], 1])])
        velocity = np.hstack([velocity, np.zeros([velocity.shape[0], 1])])

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
            # self.anchor_weights = np.array([5., 1., 5., 1., 5., 5., 1.])
            # self.anchor_weights = np.array([1., 1., 1., 1., 1., 1., 1.])

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
        u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.92,
                       (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.92,
                       (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        if np.any(u <= l):
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


def test_traj_generator(env_id='7dof-hit', seed=0):
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

    plt.hlines(-env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    plt.hlines(+env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    plt.vlines(-env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)
    plt.vlines(+env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)

    init_time = [0, 0.5, 1.0, 1.5, 2.0]
    traj_length = [25, 25, 25, 25, 100]
    colors = ['red', 'blue', 'black', 'green', 'yellow']
    cur_pos = init_c_pos
    cur_vel = init_c_vel
    j_pos = []
    j_vel = []
    for i in range(5):
        t = init_time[i]
        l = traj_length[i]
        c = colors[i]

        # weights = np.zeros(10) + 0.3
        weights = 0.3 * (2 * np.random.rand(10) - 1)
        # weights = 0.3 * np.random.normal(loc=0, scale=1.0, size=10)
        weights[4] = np.random.rand(1) + 0.5
        # weights[4] = 0.5 * np.random.normal(loc=0, scale=1.0, size=1) + 1.0
        weights[9] = 0.5 * (2 * np.random.rand(1) - 1)
        # weights[9] = 0.5 * np.random.normal(loc=0, scale=1.0, size=1)
        pos, vel = traj_gen.generate_trajectory(weights, t, cur_pos, cur_vel)
        plt.plot(pos[:l, 0] - 1.51, pos[:l, 1], color=c)
        j_pos.append(pos[:l])
        j_vel.append(vel[:l])
        cur_pos = pos[l-1]
        cur_vel = vel[l-1]
        # plt.scatter(pos[-1, 0] - 1.51, pos[-1, 1], marker='1', s=100)
        # plot_trajs(position=pos[:150], velocity=vel[:150], plot_constrs=False, plot_sampling=False, dof=3)
    plt.show()
    plot_trajs(np.concatenate(j_pos, axis=0), np.concatenate(j_vel, axis=0),
               plot_constrs=False, plot_sampling=False, dof=3)


def test_traj_optimizer(env_id='7dof-hit', seed=0):
    env_kwargs = {'check_traj': False, 'check_step': False}

    # create env
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
    env_info = env.env_info

    # init condition
    init_c_pos = np.array([0.65, 0., 0.1645])
    init_c_vel = np.array([0, 0, 0])
    init_c_acc = np.array([0, 0, 0])
    init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])
    init_j_acc = np.array([0, 0, 0, 0, 0, 0, 0])

    traj_gen = TrajectoryGenerator(env_info)
    traj_opt = TrajectoryOptimizer(env_info)

    init_time = [0, 0.5, 1.0, 1.5, 2.0]
    traj_length = [25, 25, 25, 25, 100]
    colors = ['red', 'blue', 'black', 'green', 'yellow']
    cur_c_pos = init_c_pos
    cur_c_vel = init_c_vel
    cur_c_acc = init_c_acc
    cur_j_pos = init_j_pos
    cur_j_vel = init_j_vel
    cur_j_acc = init_j_acc
    list_c_pos = []
    list_c_vel = []
    list_j_pos = []
    list_j_vel = []
    obs = env.reset()
    for i in range(5):
        init_t = init_time[i]
        traj_l = traj_length[i]

        weights = 0.1 * (2 * np.random.rand(10) - 1)
        weights[4] = (np.random.rand(1) + 0.5)
        weights[9] = (0.5 * (2 * np.random.rand(1) - 1))
        c_pos, c_vel = traj_gen.generate_trajectory(weights, init_t, cur_c_pos, cur_c_vel)
        list_c_pos.append(c_pos[:traj_l])
        list_c_vel.append(c_vel[:traj_l])
        cur_c_pos = c_pos[traj_l - 1].copy()
        cur_c_vel = c_vel[traj_l - 1].copy()
        # success, j_pos = traj_opt.optimize_trajectory(np.hstack([c_pos[:traj_l], c_vel[:traj_l]]),
        #                                               cur_j_pos, cur_j_vel, None)
        # t = np.linspace(0, j_pos.shape[0], j_pos.shape[0] + 1) * 0.02
        # f = CubicSpline(t, np.vstack([cur_j_pos, j_pos]), axis=0, bc_type=((1, cur_j_vel), (2, cur_j_acc)))
        # df = f.derivative(1)
        # ddf = f.derivative(2)

        # j_pos = f(t[1:])
        # j_vel = df(t[1:])
        # j_acc = ddf(t[1:])
        #
        # cur_j_pos = j_pos[-1]
        # cur_j_vel = j_vel[-1]
        # cur_j_acc = np.zeros(7)

        # list_j_pos.append(j_pos)
        # list_j_vel.append(j_vel)

        # for p, v in zip(j_pos, j_vel):
        #     act = np.hstack([p, v])
        #     obs_, rew, done, info = env.step(act)
        #     env.render(mode="human")

    c_pos = np.concatenate(list_c_pos)
    c_vel = np.concatenate(list_c_vel)
    success, j_pos = traj_opt.optimize_trajectory(np.hstack([c_pos, c_vel]),
                                                  cur_j_pos, cur_j_vel, None)
    t = np.linspace(0, j_pos.shape[0], j_pos.shape[0] + 1) * 0.02
    f = CubicSpline(t, np.vstack([cur_j_pos, j_pos]), axis=0, bc_type=((1, cur_j_vel), (2, cur_j_acc)))
    df = f.derivative(1)
    ddf = f.derivative(2)

    j_pos = f(t[1:])
    j_vel = df(t[1:])
    j_acc = ddf(t[1:])

    list_j_pos.append(j_pos)
    list_j_vel.append(j_vel)
    #
    for p, v in zip(j_pos, j_vel):
        act = np.hstack([p, v])
        obs_, rew, done, info = env.step(act)
        env.render(mode="human")

    plot_trajs(np.concatenate(list_j_pos, axis=0), np.concatenate(list_j_vel, axis=0),
               plot_constrs=True, plot_sampling=False, dof=7)


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
