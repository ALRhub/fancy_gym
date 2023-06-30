import time
import fancy_gym
import torch
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from test_utils import plot_trajs_j, plot_trajs_c
from test_traj_generator import TrajectoryGenerator
from test_traj_optimizer import TrajectoryOptimizer

from air_hockey_challenge.utils import world_to_robot, robot_to_world

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
                               'duration': 3.0,
                               'action_dim': 2,
                               'weights_scale': 0.4,
                               'goal_scale': 0.4,
                               'goal_offset': 1.0,
                               'disable_weights': False,
                               'disable_goal': False,
                               'relative_goal': True,
                               'auto_scale_basis': True}

traj_gen_kwargs = {'phase_generator_kwargs': phase_generator_kwargs,
                   'basis_generator_kwargs': basis_generator_kwargs,
                   'trajectory_generator_kwargs': trajectory_generator_kwargs}
traj_opt_kwargs = {}


def test_cart_agent(env_id='7dof-hit', seed=0):
    env_kwargs = {'check_traj': False, 'check_step': False}

    # create env
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
    env_info = env.env_info

    # create traj_generator and traj_optimizer
    traj_gen = TrajectoryGenerator(env_info, **traj_gen_kwargs)
    traj_opt = TrajectoryOptimizer(env_info, **traj_opt_kwargs)

    # init condition
    init_t = 0
    init_c_pos = np.array([0.65, 0., 0.1645])
    init_c_vel = np.array([0, 0, 0])
    init_c_acc = np.array([0, 0, 0])
    init_j_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    init_j_vel = np.array([0, 0, 0, 0, 0, 0, 0])
    init_j_acc = np.array([0, 0, 0, 0, 0, 0, 0])

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
                         +0.0615, -0.1882, -0.2196, -0.4647, +0.0995]),
               np.array([+0.0195, +0.5872, +0.2718, +0.7773, -0.0873,
                         -0.1504, +0.4428, -0.0797, -0.0560, -0.2223]),
               np.array([+0.6983, +1.3430, -0.2153, +0.7500, -0.6024,
                         +0.5774, -0.0146, -0.0507, +0.2924, -0.2909])]

    # interaction
    traj_length = [25, 25, 25, 75]
    obs = env.reset()
    rews = []
    jerks = []
    constrs = {'j_pos': [], 'j_vel': [], 'ee': []}
    env.render(mode="human")
    traj_gen.reset()
    traj_opt.reset()
    cur_t = init_t
    cur_c_pos = init_c_pos
    cur_c_vel = init_c_vel
    cur_c_acc = init_c_acc
    cur_j_pos = init_j_pos
    cur_j_vel = init_j_vel
    cur_j_acc = init_j_acc
    list_traj_c = []
    list_traj_j = []
    for i, l in enumerate(traj_length):
        # generate trajectory
        weight = weights[i]
        traj_c_pos, traj_c_vel = traj_gen.generate_trajectory(weight, cur_t, cur_c_pos, cur_c_vel)
        traj_c_pos, traj_c_vel = traj_c_pos[:l], traj_c_vel[:l]

        # interpolation traj c
        t = np.linspace(0, traj_c_pos.shape[0], traj_c_pos.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([cur_c_pos, traj_c_pos]), axis=0, bc_type=((1, cur_c_vel),  (2, cur_c_acc)))
        df = f.derivative(1)
        ddf = f.derivative(2)
        cur_c_pos = f(t[-1])
        cur_c_vel = df(t[-1])
        cur_c_acc = ddf(t[-1])

        # optimize trajectory
        traj_c = np.hstack([f(t[1:]), df(t[1:])])
        list_traj_c.append(traj_c.reshape([traj_c.shape[0], 2, -1]).copy())
        for s in range(int(traj_c.shape[0]/5)):
            sub_traj_c = traj_c[s * 5: (s+1) * 5]
            success, traj_j_pos = traj_opt.optimize_trajectory(sub_traj_c, cur_j_pos, cur_j_vel, None)

            # interpolation traj j
            t = np.linspace(0, traj_j_pos.shape[0], traj_j_pos.shape[0] + 1) * 0.02
            f = CubicSpline(t, np.vstack([cur_j_pos, traj_j_pos]), axis=0, bc_type=((1, cur_j_vel),  (2, cur_j_acc)))
            df = f.derivative(1)
            ddf = f.derivative(2)
            cur_j_pos = f(t[-1])
            cur_j_vel = df(t[-1])
            cur_j_acc = ddf(t[-1])

            # execute traj
            traj_j = np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
            list_traj_j.append(traj_j)
            for j in traj_j:
                act = np.hstack([j[0], j[1]])
                obs_, rew, done, info = env.step(act)
                env.render(mode="human")

                rews.append(rew)
                jerks.append(info['jerk_violation'])
                constrs['j_pos'].append(info['j_pos_violation'])
                constrs['j_vel'].append(info['j_vel_violation'])
                constrs['ee'].append(info['ee_violation'])

                if done:
                    print('Return: ', np.sum(rews))
                    print('Jerks: ', np.sum(jerks))
                    print('constr_j_pos: ', np.sum(constrs['j_pos']))
                    print('constr_j_vel: ', np.sum(constrs['j_vel']))
                    print('constr_ee: ', np.sum(constrs['ee']))
                    break

        cur_t = cur_t + 0.02 * l
        # ee_pos_world = env.base_env.get_ee()[0]
        # ee_pos_robot = world_to_robot(env_info["robot"]["base_frame"][0], ee_pos_world)[0]
        # cur_c_pos = ee_pos_robot
        # ee_vel_world = env.base_env.get_ee()[1][3:]
        # ee_vel_robot = ee_vel_world
        # cur_c_vel = ee_vel_robot

    traj_c = np.vstack(list_traj_c)
    traj_j = np.vstack(list_traj_j)
    # np.save('test_acc_action/traj_c.npy', traj_c)
    # np.save('test_acc_action/traj_j.npy', traj_j)
    print("done")


if __name__ == "__main__":
    test_cart_agent()

