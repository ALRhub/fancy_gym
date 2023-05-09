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
        pos = pos[19::20]
        vel = vel[19::20]
        acc = acc[19::20]
        jer = np.abs(jer[19::20])

    # interpolation
    tf = 0.02
    prev_pos = pos[0]
    prev_vel = vel[0]
    prev_acc = acc[0]
    prev_jer = jer[0]
    interp_pos = [prev_pos]
    interp_vel = [prev_vel]
    interp_acc = [prev_acc]
    interp_jer = [prev_jer]
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

    env = fancy_gym.make(env_id=env_id, seed=12)
    act_list = [np.array([-0.7540, -0.9243, -0.3915, -0.6968, +0.7776, +0.5865,
                          +0.2396, +0.5940, +1.0370, +0.6986, +0.2010, +0.9623]),
                np.array([-0.7244, -0.9313, -0.5614, -0.6715,  0.8473,  0.6448,
                          +0.3539,  0.7362,  1.0081,  0.8292,  0.3983,  0.9509]),
                np.array([-0.6087, -0.7917, -0.7176, -0.5665,  0.9401,  0.7882,
                          +0.5042,  0.9186,  0.9234,  0.9408,  0.5915,  0.7980])]

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        print(obs)
        if i == 0:
            env.render(mode="human")
        while True:
            # act = env.action_space.sample()
            act = act_list[1]
            obs, rew, done, info = env.step(act)

            # plot trajs
            print("weights: ", np.round(act, 2))
            if rew > -2:
                traj_pos, traj_vel = env.get_trajectory(act)
                plot_trajs(traj_pos, traj_vel, start_index=0, end_index=140, plot_sampling=False, plot_constrs=False)

            if done:
                print('Return: ', np.sum(rew))
                print('Jerks: ', np.sum(info['jerk_violation']))
                print('constr_j_pos: ', np.sum(info['constr_j_pos']))
                print('constr_j_vel: ', np.sum(info['constr_j_vel']))
                print('constr_ee: ', np.sum(info['constr_ee']))
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
            act = np.array([+0.4668, +0.2761, +0.2246, -0.0090, -0.0328, -0.5161,
                            -0.1360, -0.3141, -0.4803, -0.9457, -0.5832, -0.3209])
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
                print('constr_j_pos: ', np.sum(info['constr_j_pos']))
                print('constr_j_vel: ', np.sum(info['constr_j_vel']))
                print('constr_ee: ', np.sum(info['constr_ee']))
            if done:
                step = np.linspace(1, 150, 150)
                for pos, vel in zip(pos_list, vel_list):
                    plt.subplot(1, 2, 1)
                    plt.plot(step, pos[:, 0])

                    plt.subplot(1, 2, 2)
                    plt.plot(step, vel[:, 0])

                plt.show()
                break


if __name__ == "__main__":
    # test_baseline(env_id='3dof-hit-sparse', iteration=1)
    # test_env(env_id="3dof-hit-sparse", iteration=10)
    test_mp_env(env_id="3dof-hit-sparse-prodmp", seed=1, iteration=3)
    # test_replan_env(env_id="3dof-hit-sparse-prodmp-replan", seed=1, iteration=3)

