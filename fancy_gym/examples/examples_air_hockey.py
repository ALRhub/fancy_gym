import os
import time
import numpy as np

import fancy_gym
from baseline.baseline_agent.baseline_agent import build_agent


def plot_trajs(position, velocity):
    import matplotlib.pyplot as plt

    pos = position
    vel = np.diff(pos, n=1, axis=0) / 0.02
    acc = np.diff(vel, n=1, axis=0) / 0.02
    jer = np.diff(acc, n=1, axis=0) / 0.02

    pos = pos[:-3]
    vel = vel[:-2]
    acc = acc[:-1]
    jer = jer

    time = np.linspace(0, 2.4, pos.shape[0])
    constr_j_pos = [[-2.9, +2.9], [-1.8, +1.8], [-2.0, +2.0]]
    constr_j_vel = [[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]]
    # plt.subplots(3, 2)
    for k in range(3):
        plt.subplot(3, 4, 4 * k + 1)
        plt.plot(time, pos[:, k])
        plt.hlines(constr_j_pos[k], xmin=time[0], xmax=time[-1])

        plt.subplot(3, 4, 4 * k + 2)
        plt.plot(time, vel[:, k])
        # plt.plot(time, velocity[:-3, k])
        plt.hlines(constr_j_vel[k], xmin=time[0], xmax=time[-1])

        plt.subplot(3, 4, 4 * k + 3)
        plt.plot(time, acc[:, k])

        plt.subplot(3, 4, 4 * k + 4)
        plt.plot(time, jer[:, k])
    plt.show()


def test_baseline(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    agent = build_agent(env_info)
    for i in range(iteration):
        rews = []
        jerks = []
        constrs = {'j_pos': [], 'j_vel': [], 'ee': []}

        j_pos = []
        j_vel = []

        agent.reset()
        obs = env.reset()
        env.render(mode="human")
        while True:
            act = agent.draw_action(obs).reshape([-1])
            obs_, rew, done, info = env.step(act)
            env.render(mode="human")

            j_pos.append(act[:3])
            j_vel.append(act[3:])

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
                # plot_trajs(np.array(j_pos), np.array(j_vel))
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

    for i in range(iteration):
        print("*"*20, i, "*"*20)
        obs = env.reset()
        if i == 0:
            env.render(mode="human")
        while True:
            act = env.action_space.sample()
            # act = np.array([0.1, 0.2, 0.1, 0.1, -0.1, -0.2, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1]) * -2
            obs, rew, done, info = env.step(act)

            # plot trajs
            print("weights: ", np.round(act, 2))
            if rew > -2:
                traj_pos, traj_vel = env.get_trajectory(act)
                # plot_trajs(traj_pos, traj_vel)

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
    # test_baseline(env_id='3dof-hit-sparse', iteration=10)
    # test_env(env_id="3dof-hit-sparse", iteration=10)
    test_mp_env(env_id="3dof-hit-sparse-promp", seed=1, iteration=1000)
    # test_mp()
