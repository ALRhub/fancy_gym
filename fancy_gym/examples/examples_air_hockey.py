import os
import time
import numpy as np

import fancy_gym
from baseline.baseline_agent.baseline_agent import build_agent

import matplotlib.pyplot as plt
import scipy


def plot_trajs(position, velocity, jerk):
    dt = 0.02
    pos = position
    vel = np.diff(pos, n=1, axis=0) / dt
    acc = np.diff(vel, n=1, axis=0) / dt
    jer = np.diff(acc, n=1, axis=0) / dt

    pos = position[:-9]  # 141 steps for 2801 interp steps
    vel = velocity[:-9]
    acc = acc[:-7]
    jer = jer[:-6]
    jerk = jerk[:-9]

    # interp pos
    tf = dt
    prev_pos = position[0]
    prev_vel = velocity[0]
    prev_acc = np.array([0, 0, 0])
    interp_pos = [prev_pos]
    interp_vel = [prev_vel]
    interp_acc = [prev_acc]
    for i in range(position.shape[0] - 10):
        coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
        reg = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-4, 0], [0, 0, 0, 0.001]])
        # coef = coef + reg
        results = np.vstack([prev_pos, position[i+1], prev_vel, velocity[i+1]])
        A = scipy.linalg.block_diag(*[coef] * 3)
        y = results.reshape(-1, order='F')
        weights = np.linalg.solve(A, y).reshape(3, 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

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

    step = np.linspace(dt, dt * pos.shape[0], pos.shape[0])
    constr_j_pos = [[-2.9, +2.9], [-1.8, +1.8], [-2.0, +2.0]]
    constr_j_vel = [[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]]
    constr_j_jerk = [[0, 1e4]] * 3

    interp_pos = np.array(interp_pos)
    interp_vel = np.array(interp_vel)
    interp_acc = np.array(interp_acc)
    interp_step = np.linspace(dt, dt * pos.shape[0], interp_pos.shape[0])

    s = 130
    ss = 20 * s
    e = 141
    ee = 20 * (e - 1) + 1
    for k in range(3):
        plt.subplot(3, 5, 5 * k + 1)
        if k == 0:
            plt.title("mp_pos vs. interp_pos")
        plt.plot(step[s:e], pos[s:e, k])
        plt.plot(interp_step[ss:ee], interp_pos[ss:ee, k])
        # plt.hlines(constr_j_pos[k], xmin=step[0], xmax=step[-1], colors="r")

        plt.subplot(3, 5, 5 * k + 2)
        if k == 0:
            plt.title("mp_vel vs. interp_vel")
        plt.plot(step[s:e], vel[s:e, k])
        plt.plot(interp_step[ss:ee], interp_vel[ss:ee, k])
        # plt.hlines(constr_j_vel[k], xmin=step[0], xmax=step[-1], colors="r")

        plt.subplot(3, 5, 5 * k + 3)
        if k == 0:
            plt.title("mp_acc vs. interp_acc")
        plt.plot(interp_step[ss:ee], interp_acc[ss:ee, k])
        plt.plot(step[s:e], acc[s:e, k])

        plt.subplot(3, 5, 5 * k + 4)
        if k == 0:
            plt.title("mp_jerk")
        plt.plot(step, jer[:, k])

        plt.subplot(3, 5, 5 * k + 5)
        if k == 0:
            plt.title("interp_jerk")
        plt.plot(step, jerk[:, k])
        plt.hlines(constr_j_jerk[k], xmin=step[0], xmax=step[-1], colors='r')
    plt.show()


def plot_jerks(jerks):
    time = np.linspace(0, 3, jerks.shape[0])
    for d in range(3):
        plt.subplot(3, 1, d+1)
        plt.plot(time, jerks[:, d])
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
    env = fancy_gym.make(env_id=env_id, seed=seed)

    act_list = [np.array([+0.4668, +0.2761, +0.2246, -0.0090, -0.0328, -0.5161,
                          -0.1360, -0.3141, -0.4803, -0.9457, -0.5832, -0.3209]),
                np.array([+0.3490, +0.0597, +0.1681, +0.2891, -0.3729, -0.6172,
                          -0.4291, -0.3400, -0.5428, -0.8549, -0.5452, -0.2657]),
                np.array([+0.0796, +0.1585, +0.1650, +0.2431, -0.5435, -0.5349,
                          -0.3300, -0.3566, -0.5377, -1.0708, -0.5976, -0.3642])]

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        # if i == 0:
        #     env.render(mode="human")
        while True:
            act = env.action_space.sample()
            # act = np.hstack([act, act, act])
            act = act_list[0]
            obs, rew, done, info = env.step(act)

            # plot trajs
            print("weights: ", np.round(act, 2))
            if rew > -2:
                traj_pos, traj_vel = env.get_trajectory(act)
                plot_trajs(traj_pos, traj_vel, np.array(info['jerk']))
                # plot_jerks(np.array(info['jerk']))

            if done:
                print('Return: ', np.sum(rew))
                print('Jerks: ', np.sum(info['jerk_violation']))
                print('constr_j_pos: ', np.sum(info['constr_j_pos']))
                print('constr_j_vel: ', np.sum(info['constr_j_vel']))
                print('constr_ee: ', np.sum(info['constr_ee']))
                break


def test_mp():
    import torch as tc
    import matplotlib.pyplot as plt

    from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
    from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
    from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator

    for phase_type in ['linear', 'exp']:
        phase_kwargs = {'phase_generator_type': phase_type, 'tau': 2.4, 'delay': 0.0}
        phase_gen = get_phase_generator(**phase_kwargs)
        duration = 2.4
        dt = 0.02
        times = tc.linspace(0, duration, round(duration/dt)+1)
        phase = phase_gen.phase(times)
        # plt.plot(steps, phase)
        # plt.show()

        for basis_type in ['rbf', 'zero_rbf', 'prodmp']:
            if basis_type == 'prodmp':
                continue
            basis_kwargs = {'basis_generator_type': basis_type,
                            'num_basis': 2,
                            'basis_bandwidth_factor': 3}
            if basis_type == 'zero_rbf':
                basis_kwargs['num_basis_zero_start'] = 1
                basis_kwargs['num_basis_zero_goal'] = 1
            basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_kwargs)
            basis = basis_gen.basis(times)
            # for i in range(basis.shape[-1]):
            #     plt.plot(times, basis[:, i])
            # plt.show()

            for traj_type in ['ProMP', 'DMP', 'ProDMP']:
                if traj_type == 'ProDMP' or traj_type == 'DMP':
                    continue
                traj_kwargs = {'trajectory_generator_type': traj_type,
                               'action_dim': 1}
                traj_gen = get_trajectory_generator(basis_generator=basis_gen, **traj_kwargs)
                init_time = np.array(0)
                init_pos = np.array([0.5])
                init_vel = np.array([0])
                traj_gen.set_initial_conditions(init_time, init_pos, init_vel)
                traj_gen.set_duration(duration, dt)

                weights = np.random.random(basis_kwargs['num_basis'])
                traj_gen.set_params(weights)

                pos = traj_gen.get_traj_pos()[:, -1].numpy()
                vel = traj_gen.get_traj_vel()[:, -1].numpy()
                pos = np.hstack([init_pos, pos])
                vel = np.hstack([init_vel, vel])

                plt.subplot(2, 1, 1)
                plt.plot(times, pos)

                plt.subplot(2, 1, 2)
                plt.plot(times, vel)

                plt.suptitle(traj_type + '_' + basis_type + '_' + phase_type)
                plt.show()


def test_learn_mp():
    pass


if __name__ == "__main__":
    # test_baseline(env_id='3dof-hit-sparse', iteration=1)
    # test_env(env_id="3dof-hit-sparse", iteration=10)
    test_mp_env(env_id="3dof-hit-sparse-promp", seed=1, iteration=1)
    # test_mp()
