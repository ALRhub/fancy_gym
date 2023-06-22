import os
import time
import random
import numpy as np

from test_utils import plot_trajs
import fancy_gym

import matplotlib.pyplot as plt


def test_env(env_id="3dof-hit", seed=0, iteration=5):
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardV2',
                  'check_step': False, 'check_traj': False}
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)

    for i in range(iteration):
        rews = []
        j_pos = []
        j_vel = []

        obs = env.reset()
        env.render(mode="human")
        step = 0
        while True:
            act = env.action_space.sample()
            obs_, rew, done, info = env.step(act)
            env.render(mode='human')

            rews.append(rew)
            j_pos.append(act[:3])
            j_vel.append(act[3:])

            step += 1
            print(step)

            if done:
                print('Return: ', np.sum(rews))
                print('num_ee_x_violation: ', info['num_ee_x_violation'])
                print('num_ee_y_violation: ', info['num_ee_y_violation'])
                print('num_ee_z_violation: ', info['num_ee_z_violation'])
                print('num_jerk_violation: ', info['num_jerk_violation'])
                print('num_j_pos_violation: ', info['num_j_pos_violation'])
                print('num_j_vel_violation: ', info['num_j_vel_violation'])
                # plot_trajs(np.array(j_pos), np.array(j_vel))
                break


def test_mp_env(env_id="3dof-hit-promp", seed=0, iteration=5, plot_result=True):
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardWaitPuck',
                  'check_step': True, 'check_traj':  True, 'check_traj_length': [-1],
                  'early_stop': True, 'wait_puck': True}
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)

    _act = np.array([6.49932e-01, 2.05280e-05, 0.1645])[:2]
    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        if i == 0:
            env.render(mode="human")
        while True:
            # act = env.action_space.sample()
            # act = act_list[0]
            # act = np.ones(8) * 0.1
            # act = (2 * np.random.rand(28) - 1) * 0.1
            # act = np.random.rand(8) * 0.5
            act = np.array([0.4758, 0.7876, 0.4221, 0.5428, 0.4740,
                            0.3129, 0.0619, 0.4272, 0.0162, 0.0826])

            # plot trajs
            if plot_result:
                print("weights: ", np.round(act, 2))
                env.traj_gen.show_scaled_basis(plot=True)
                current_pos, current_vel = env.current_pos, env.current_vel
                traj_pos, traj_vel = env.get_trajectory(act)
                traj_pos = np.vstack([current_pos, traj_pos])
                traj_vel = np.vstack([current_vel, traj_vel])
                plot_trajs(traj_pos, traj_vel, start_index=0, end_index=150, plot_sampling=True, plot_constrs=True, dof=7)

            obs, rew, done, info = env.step(act)

            if done:
                print('Return: ', np.sum(rew))
                print('jerk_violation: ', np.sum(info['jerk_violation']))
                print('j_pos_violation: ', np.sum(info['j_pos_violation']))
                print('j_vel_violation: ', np.sum(info['j_vel_violation']))
                print('ee_violation: ', np.sum(info['ee_violation']))
                break


def test_mp_replan_env(env_id="3dof-hit-prodmp-replan", seed=0, iteration=5, plot_result=True):
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardV2',
                  'check_step': False, 'check_traj': False, 'check_traj_length': [25, 25, 25, 75]}
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        done = False
        if i == 0:
            env.render(mode="human")

        current_pos, current_vel = env.current_pos, env.current_vel
        pos_list, vel_list = [current_pos], [current_vel]
        pos_list, vel_list = [], []
        if '3dof' in env_id:
            act = np.array([6.49932e-01, 2.05280e-05, 1.00000e-01])[:2]
        else:
            act = np.array([6.49932e-01, 2.05280e-05, 0.1645])[:2]

        while True:
            # act = env.action_space.sample() * 0.01
            # init_j_pos = np.array([-1.15570, +1.30024, +1.44280])
            # act = np.array([-0.6244, -0.6889, -0.2778, -0.6943,  0.6887,  0.5214,
            #                 +0.1311,  0.6478,  0.8111,  0.4709, -0.0475,  0.3196])
            # while True:
            #     act = np.random.rand(2)
            #     break
            act = act + 0.05
            pos, vel = env.get_trajectory(act)
            pos_list.append(pos), vel_list.append(vel)
            obs, rew, done, info = env.step(act)
            print(info['compute_time_ms'])
            if True:
                print('*'*20, 'segment', len(pos_list)-1, '*'*20)
                print('Return: ', np.sum(rew))
                print('Jerks: ', np.sum(info['jerk_violation']))
                print('constr_j_pos: ', np.sum(info['j_pos_violation']))
                print('constr_j_vel: ', np.sum(info['j_vel_violation']))
                print('constr_ee: ', np.sum(info['ee_violation']))

            if done:
                step = np.linspace(0.001, 3, 3000) - 0.5
                colors = ['r', 'g', 'b', 'y', 'black']
                idx = 0
                for pos, vel in zip(pos_list, vel_list):
                    step += 0.5

                    # plt.subplot(1, 2, 1)
                    # plt.plot(step, pos[:, 0], color=colors[idx])
                    #
                    # plt.subplot(1, 2, 2)
                    # plt.plot(step, vel[:, 0], color=colors[idx])

                    plt.plot(pos[:, 0], pos[:, 1], color=colors[idx])

                    idx += 1
                plt.show()
                break


if __name__ == "__main__":
    # test_env(env_id="7dof-hit", iteration=10)
    test_mp_env(env_id="7dof-hit-cart-promp", seed=1, iteration=10, plot_result=False)
    # test_mp_replan_env(env_id="7dof-hit-cart-promp-replan", seed=1, iteration=3, plot_result=False)

