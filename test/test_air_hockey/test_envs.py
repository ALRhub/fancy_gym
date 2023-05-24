import os
import time
import numpy as np

import fancy_gym
from baseline.baseline_agent.baseline_agent import build_agent

import matplotlib.pyplot as plt
import scipy


def plot_trajs(position, velocity, start_index=0, end_index=100, plot_sampling=True, plot_constrs=True):
    if plot_sampling:
        dt = 0.001
    else:
        dt = 0.02
    pos = position
    vel = velocity
    acc = np.diff(vel, n=1, axis=0, append=np.zeros([1, 3])) / dt
    jer = np.diff(acc, n=1, axis=0, append=np.zeros([1, 3])) / dt
    jer = np.abs(jer)

    # down sampling
    if plot_sampling:
        pos = pos[::20]
        vel = vel[::20]
        acc = acc[::20]
        jer = np.abs(jer[::20])

    # interpolation
    tf = 0.02
    prev_pos = pos[0]
    prev_vel = 0 * vel[0]
    prev_acc = 0 * acc[0]
    interp_pos = [prev_pos]
    interp_vel = [prev_vel]
    interp_acc = [prev_acc]
    interp_jer = []
    for i in range(pos.shape[0] - 1):
        coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
        results = np.vstack([prev_pos, pos[i+1], prev_vel, vel[i+1]])
        A = scipy.linalg.block_diag(*[coef] * 3)
        y = results.reshape(-1, order='F')
        weights = np.linalg.solve(A, y).reshape(3, 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        interp_jer.append(np.abs(weights_dd[:, 1]) + np.abs(weights_dd[:, 0] - prev_acc) / 0.001)

        prev_pos = np.polynomial.polynomial.polyval(tf, weights.T)
        prev_vel = np.polynomial.polynomial.polyval(tf, weights_d.T)
        prev_acc = np.polynomial.polynomial.polyval(tf, weights_dd.T)

        for t in np.linspace(0.001, 0.02, 20):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            interp_pos.append(q)
            interp_vel.append(qd)
            interp_acc.append(qdd)

    constr_j_pos = [[-2.9, +2.9], [-1.8, +1.8], [-2.0, +2.0]]
    constr_j_vel = [[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]]
    constr_j_jerk = [[0, 1e4]] * 3

    interp_pos = np.array(interp_pos)
    interp_vel = np.array(interp_vel)
    interp_acc = np.array(interp_acc)
    interp_jer = np.array(interp_jer)

    step = np.linspace(0.02, 3, pos.shape[0])
    interp_step = np.linspace(0.02, 3, interp_pos.shape[0])

    s = start_index
    ss = 20 * s
    e = end_index
    ee = 20 * (e - 1) + 1
    for d in range(3):
        plt.subplot(3, 4, 4 * d + 1)
        if d == 0:
            plt.title("mp_pos vs. interp_pos")
        plt.plot(step[s:e], pos[s:e, d], color='blue')
        plt.plot(interp_step[ss:ee], interp_pos[ss:ee, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_pos[d], xmin=step[s], xmax=step[e], colors="r")

        plt.subplot(3, 4, 4 * d + 2)
        if d == 0:
            plt.title("mp_vel vs. interp_vel")
        plt.plot(step[s:e], vel[s:e, d], color='blue')
        plt.plot(interp_step[ss:ee], interp_vel[ss:ee, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_vel[d], xmin=step[s], xmax=step[e], colors="r")

        plt.subplot(3, 4, 4 * d + 3)
        if d == 0:
            plt.title("mp_acc vs. interp_acc")
        plt.plot(interp_step[ss:ee], interp_acc[ss:ee, d], color='green')
        plt.plot(step[s:e], acc[s:e, d], color='blue')

        plt.subplot(3, 4, 4 * d + 4)
        if d == 0:
            plt.title("mp_jerk vs. interp_jerk")
        plt.plot(step[s:e], jer[s:e, d], color='blue')
        plt.plot(step[s:e], interp_jer[s:e, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_jerk[d], xmin=step[s], xmax=step[e], colors='r')
    plt.show()


def test_baseline(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    agent = build_agent(env_info)
    for i in range(iteration):
        rews = []
        jerks = []
        constrs = {'j_pos': [], 'j_vel': [], 'ee': []}
        traj_jerk = []

        j_pos = []
        j_vel = []

        agent.reset()
        obs = env.reset()
        # env.render(mode="human")
        while True:
            act = agent.draw_action(obs).reshape([-1])
            obs_, rew, done, info = env.step(act)
            # env.render(mode="human")

            j_pos.append(act[:3])
            j_vel.append(act[3:])
            traj_jerk.append(info["jerk"])

            rews.append(rew)
            jerks.append(info['jerk_violation'])
            constrs['j_pos'].append(info['constr_j_pos'])
            constrs['j_vel'].append(info['constr_j_vel'])
            constrs['ee'].append(info['constr_ee'])

            if done:
                print('Return: ', np.sum(rews))
                print('Jerks: ', np.sum(jerks))
                print('constr_j_pos: ', np.sum(constrs['j_pos']))
                print('constr_j_vel: ', np.sum(constrs['j_vel']))
                print('constr_ee: ', np.sum(constrs['ee']))
                plot_trajs(np.array(j_pos), np.array(j_vel), np.array(traj_jerk))
                break


def test_env(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    for i in range(iteration):
        rews = []
        jerks = []
        constrs = {'j_pos': [], 'j_vel': [], 'ee': []}

        obs = env.reset()
        # env.render(mode="human")
        while True:
            act = env.action_space.sample()
            obs_, rew, done, info = env.step(act)
            # env.render(mode='human')

            rews.append(rew)
            jerks.append(info['jerk_violation'])
            constrs['j_pos'].append(info['constr_j_pos'])
            constrs['j_vel'].append(info['constr_j_vel'])
            constrs['ee'].append(info['constr_ee'])

            if done:
                print('Return: ', np.sum(rews))
                print('Jerks: ', np.sum(jerks))
                print('constr_j_pos: ', np.sum(constrs['j_pos']))
                print('constr_j_vel: ', np.sum(constrs['j_vel']))
                print('constr_ee: ', np.sum(constrs['ee']))
                break


def test_mp_env(env_id="3dof-hit-promp", seed=0, iteration=5):
    import random
    random.seed(seed)
    np.random.seed(seed)

    env_kwargs = {'dt': 0.001, 'reward_function': 'HitSparseRewardV1'}
    env = fancy_gym.make(env_id=env_id, seed=12, **env_kwargs)

    # ProMP samples
    act_list = [np.array([+0.4668, +0.2761, +0.2246, -0.0090, -0.0328, -0.5161,
                          -0.1360, -0.3141, -0.4803, -0.9457, -0.5832, -0.3209]),
                np.array([+0.3490, +0.0597, +0.1681, +0.2891, -0.3729, -0.6172,
                          -0.4291, -0.3400, -0.5428, -0.8549, -0.5452, -0.2657]),
                np.array([+0.0796, +0.1585, +0.1650, +0.2431, -0.5435, -0.5349,
                          -0.3300, -0.3566, -0.5377, -1.0708, -0.5976, -0.3642])]

    # ProDMP samples
    # act_list = [np.array([-0.6244, -0.6889, -0.2778, -0.6943,  0.6887,  0.5214,
    #                       +0.1311,  0.6478,  0.8111,  0.4709, -0.0475,  0.3196]),
    #             np.array([-0.6474, -0.7177, -0.2084, -0.7114,  0.6966,  0.5063,
    #                       +0.1093,  0.6917,  0.7944,  0.4167, -0.1352,  0.2618]),
    #             np.array([-0.7244, -0.9313, -0.5614, -0.6715,  0.8473,  0.6448,
    #                       +0.3539,  0.7362,  1.0081,  0.8292,  0.3983,  0.9509]),
    #             np.array([-0.6087, -0.7917, -0.7176, -0.5665,  0.9401,  0.7882,
    #                       +0.5042,  0.9186,  0.9234,  0.9408,  0.5915,  0.7980])]

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        if i == 0:
            env.render(mode="human")
        while True:
            act = env.action_space.sample()
            act = act_list[i]
            obs, rew, done, info = env.step(act)

            # plot trajs
            print("weights: ", np.round(act, 2))
            if rew > -2:
                obs = env.reset()
                env.traj_gen.show_scaled_basis(plot=True)
                traj_pos, traj_vel = env.get_trajectory(act)
                current_pos, current_vel = env.current_pos, env.current_vel
                traj_pos = np.vstack([current_pos, traj_pos])
                traj_vel = np.vstack([current_vel, traj_vel])
                plot_trajs(traj_pos, traj_vel, start_index=0, end_index=150, plot_sampling=True, plot_constrs=True)

            if done:
                print('Return: ', np.sum(rew))
                print('jerk_violation: ', np.sum(info['jerk_violation']))
                print('j_pos_violation: ', np.sum(info['j_pos_violation']))
                print('j_vel-violation: ', np.sum(info['j_vel_violation']))
                print('ee_violation: ', np.sum(info['ee_violation']))
                break


def test_replan_env(env_id="3dof-hit-prodmp-replan", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        done = False
        env.render(mode="human")
        pos_list = []
        vel_list = []
        while True:
            act = env.action_space.sample()
            act = np.array([-0.6244, -0.6889, -0.2778, -0.6943,  0.6887,  0.5214,
                            +0.1311,  0.6478,  0.8111,  0.4709, -0.0475,  0.3196])
            pos, vel = env.get_trajectory(act)
            pos_list.append(pos), vel_list.append(vel)
            obs, rew, done, info = env.step(act)

            # plot trajs
            # print("weights: ", np.round(act, 2))
            # if rew > -2:
            #     traj_pos, traj_vel = env.get_trajectory(act)
            #     plot_trajs(traj_pos, traj_vel, start_index=0, end_index=140, plot_sampling=False, plot_constrs=False)

            if True:
                print('Return: ', np.sum(rew))
                print('Jerks: ', np.sum(info['jerk_violation']))
                print('constr_j_pos: ', np.sum(info['j_pos_violation']))
                print('constr_j_vel: ', np.sum(info['j_vel_violation']))
                print('constr_ee: ', np.sum(info['ee_violation']))
            if done:
                step = np.linspace(0.02, 3, 150) - 1
                colors = ['r', 'g', 'b']
                idx = 0
                for pos, vel in zip(pos_list, vel_list):
                    step += 1

                    plt.subplot(1, 2, 1)
                    plt.plot(step, pos[:, 0], color=colors[idx])

                    plt.subplot(1, 2, 2)
                    plt.plot(step, vel[:, 0], color=colors[idx])

                    idx += 1
                plt.show()
                break


if __name__ == "__main__":
    # test_baseline(env_id='3dof-hit-sparse', iteration=1)
    # test_env(env_id="3dof-hit-sparse", iteration=10)
    test_mp_env(env_id="3dof-hit-promp", seed=1, iteration=3)
    # test_replan_env(env_id="3dof-hit-sparse-prodmp-replan", seed=1, iteration=3)

