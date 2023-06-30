import torch
import numpy as np
from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator

import fancy_gym
import matplotlib.pyplot as plt
from test_utils import plot_trajs_j, plot_trajs_c

from scipy.interpolate import CubicSpline


class TrajectoryGenerator:
    def __init__(self, env_info, **kwargs):
        self.env_info = env_info

        # init condition
        self.init_t = np.array([0])
        self.init_c_pos = np.array([0.65, 0.0, 0.0])
        self.init_c_vel = np.array([0., 0., 0.])
        if '3dof' in env_info['env_name']:
            self.n_joints = 3
            self.init_c_pos[2] = 0.1000
        else:
            self.n_joints = 7
            self.init_c_pos[2] = 0.1645

        # boundary_condition
        self.prev_t = self.init_t
        self.prev_c_pos = self.init_c_pos
        self.prev_c_vel = self.init_c_vel

        # setting for mp_pytorch
        self.dt = 0.02
        self.duration = 3.0
        phase_generator_kwargs = kwargs['phase_generator_kwargs']
        basis_generator_kwargs = kwargs['basis_generator_kwargs']
        trajectory_generator_kwargs = kwargs['trajectory_generator_kwargs']
        phase_gen = get_phase_generator(**phase_generator_kwargs)
        basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_generator_kwargs)
        self.traj_gen = get_trajectory_generator(basis_generator=basis_gen, **trajectory_generator_kwargs)

        # basis lookup table
        # todo: store basis as lookup table
        pass

    def reset(self):
        self.traj_gen.reset()
        self.traj_gen.set_duration(self.duration, self.dt)

        # reset boundary condition
        self.prev_t = self.init_t
        self.prev_c_pos = self.init_c_pos
        self.prev_c_vel = self.init_c_vel

    @torch.no_grad()
    def generate_trajectory(self, weights, prev_t=None, prev_c_pos=None, prev_c_vel=None):
        prev_t = self.prev_t if prev_t is None else prev_t
        prev_c_pos = self.prev_c_pos if prev_c_pos is None else prev_c_pos
        prev_c_vel = self.prev_c_vel if prev_c_vel is None else prev_c_vel

        # self.traj_gen.reset()
        self.traj_gen.set_params(weights)
        self.traj_gen.set_initial_conditions(prev_t, prev_c_pos[:2], prev_c_vel[:2])
        self.traj_gen.set_duration(self.duration, self.dt)

        # get position and velocity
        traj_c_pos = get_numpy(self.traj_gen.get_traj_pos())
        traj_c_vel = get_numpy(self.traj_gen.get_traj_vel())

        if self.dt == 0.001:
            traj_c_pos = traj_c_pos[19::20].copy()
            traj_c_vel = traj_c_vel[19::20].copy()

        if self.n_joints == 3:
            traj_c_pos = np.hstack([traj_c_pos, 0.1000 * np.ones([traj_c_pos.shape[0], 1])])
        else:
            traj_c_pos = np.hstack([traj_c_pos, 0.1645 * np.ones([traj_c_pos.shape[0], 1])])
        traj_c_vel = np.hstack([traj_c_vel, np.zeros([traj_c_vel.shape[0], 1])])

        return traj_c_pos, traj_c_vel


def get_numpy(x: torch.Tensor):
    """
    Returns numpy array from torch tensor
    Args:
        x:

    Returns:

    """
    return x.detach().cpu().numpy()


def sample_weights(gaussian=True):
    if gaussian:
        weights = 0.3 * np.random.normal(loc=0, scale=1.0, size=10)
        weights[4] = 0.5 * np.random.normal(loc=0, scale=1.0, size=1) + 1.0
        weights[9] = 0.5 * np.random.normal(loc=0, scale=1.0, size=1)
    else:
        weights = 0.3 * (2 * np.random.rand(10) - 1)
        weights[4] = np.random.rand(1) + 0.5
        weights[9] = 0.5 * (2 * np.random.rand(1) - 1)
    return weights


def test_traj_generator(env_id='7dof-hit', seed=0, traj_gen_kwargs=None):
    # create env
    env_kwargs = {'check_traj': False, 'check_step': False}
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
    env_info = env.env_info

    # create trajectory generator
    traj_gen = TrajectoryGenerator(env_info, **traj_gen_kwargs)

    # weights
    weights = [np.array([+0.7409, -0.7242, +1.0680, -0.7881, -0.3357,
                         +0.1106, +0.1621, +1.7260, -0.5713, +0.0182]),
               np.array([+0.8588, -0.1287, +1.2832, -0.9561, +0.2516,
                         +0.0689, +0.5649, +0.9032, +1.5058, -0.3095]),
               np.array([-0.0678, +0.2216, -0.2608, +0.4018, +0.3771,
                         +0.1719, +0.1977, -0.4107, +0.9944, +0.0578]),
               np.array([+0.8626, +0.4412, +1.4282, -3.4969, +0.1617,
                         -0.1037, +0.3568, -0.2880, +2.9234, -0.3282])]

    weights = [np.array([+0.6314, +0.4084, -0.0947, +0.3854, -0.6449,
				         -0.0692, -0.6694, -0.3800, -0.7310, +0.4509]),
               np.array([+0.8584, +1.0769, -0.2566, +0.4282, +0.4002,
                         +0.0615, -0.1882, -0.2196, -0.4647,  0.0995]),
               np.array([+0.0195,  0.5872,  0.2718,  0.7773, -0.0873,
                         -0.1504,  0.4428, -0.0797, -0.0560, -0.2223]),
               np.array([+0.6983,  1.3430, -0.2153,  0.7500,  -0.6024,
                        +0.5774, -0.0146, -0.0507,  0.2924, -0.2909])]

    # plot
    plt.hlines(-env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    plt.hlines(+env_info['table']['width'] / 2, xmin=-1.1, xmax=+1.1)
    plt.vlines(-env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)
    plt.vlines(+env_info['table']['length'] / 2, ymin=-0.6, ymax=+0.6)

    replan_time = [0, 0.5, 1.0, 1.5]
    traj_length = [25, 25, 25, 75]
    colors = ['red', 'green', 'blue', 'yellow']
    list_c_pos = []
    list_c_vel = []
    traj_gen.reset()
    cur_c_pos = np.array([0.65, 0., 0.1645])
    cur_c_vel = np.array([0, 0, 0])
    cur_c_acc = np.array([0, 0, 0])
    list_c_pos.append(cur_c_pos)
    list_c_vel.append(cur_c_vel)
    for i, [t, l] in enumerate(zip(replan_time, traj_length)):
        weight = weights[i]
        cur_t = t
        traj_c_pos, traj_c_vel = traj_gen.generate_trajectory(weight, cur_t, cur_c_pos, cur_c_vel)
        traj_c_pos, traj_c_vel = traj_c_pos[:l], traj_c_vel[:l]

        t = np.linspace(0, traj_c_pos.shape[0], traj_c_pos.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([cur_c_pos, traj_c_pos]), axis=0, bc_type=((1, cur_c_vel),  (2, cur_c_acc)))
        df = f.derivative(1)
        ddf = f.derivative(2)

        # plot_trajs_j(traj_c_pos, traj_c_vel, 0, 99, False, False, 3)
        # plt.plot(f(t[1:])[:, 0] - 1.51, f(t[1:])[:, 1], color=colors[i])
        plt.plot(f(t[1:])[:, 0] - 1.51, f(t[1:])[:, 1], color=colors[i])
        # cur_c_pos = f(t[-1])
        # cur_c_vel = df(t[-1])
        # cur_c_acc = ddf(t[-1])
        #
        # list_c_pos.append(f(t[1:]))
        # list_c_vel.append(df(t[1:]))
        #
        cur_c_pos = traj_c_pos[-1]
        cur_c_vel = traj_c_vel[-1]
        cur_c_acc = ddf(t[-1])

        list_c_pos.append(traj_c_pos)
        list_c_vel.append(traj_c_vel)
        # plt.scatter(pos[-1, 0] - 1.51, pos[-1, 1], marker='1', s=100)
        # plot_trajs(position=pos[:150], velocity=vel[:150], plot_constrs=False, plot_sampling=False, dof=3)
    plt.show()
    # plot_trajs_j(np.vstack(list_c_pos), np.vstack(list_c_vel), 0, 140, False, False, 3)

phase_generator_kwargs = {'phase_generator_type': 'exp',
                          'delay': 0,
                          'tau': 3.0,
                          'alpha_phase': 3}
basis_generator_kwargs = {'basis_generator_type': 'prodmp',
                          'num_basis': 4,
                          'alpha': 15,
                          'basis_bandwidth_factor': 3.0}
trajectory_generator_kwargs = {'trajectory_generator_type': 'prodmp',
                               'duration': 3.0,
                               'action_dim': 2,
                               'weights_scale': 0.4,
                               'goal_scale': 0.4,
                               'goal_offset': 1.0,
                               'disable_weights': False,
                               'disable_goal': False,
                               'relative_goal': True,
                               'auto_scale_basis': True}

if __name__ == '__main__':
    gen_kwargs = {'phase_generator_kwargs': phase_generator_kwargs,
                  'basis_generator_kwargs': basis_generator_kwargs,
                  'trajectory_generator_kwargs': trajectory_generator_kwargs}
    test_traj_generator(env_id='7dof-hit', seed=0, traj_gen_kwargs=gen_kwargs)
