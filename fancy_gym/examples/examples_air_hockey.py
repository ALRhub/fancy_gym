import numpy as np

import fancy_gym
from baseline.baseline_agent.baseline_agent import build_agent


def test_baseline(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    agent = build_agent(env_info)
    for i in range(iteration):
        rews = []
        jerks = []
        constrs = {'j_pos': [], 'j_vel': [], 'ee': []}

        agent.reset()
        obs = env.reset()
        env.render(mode="human")
        while True:
            act = agent.draw_action(obs).reshape([-1])
            obs_, rew, done, info = env.step(act)
            env.render(mode="human")

            rews.append(rew)
            jerks.append(info['jerk'])
            constrs['j_pos'].append(np.any(info['constraints_value']['joint_pos_constr'] > 0).astype(int))
            constrs['j_vel'].append(np.any(info['constraints_value']['joint_vel_constr'] > 0).astype(int))
            constrs['ee'].append(np.any(info['constraints_value']['ee_constr'] > 0).astype(int))

            if done:
                print('Return: ', np.sum(rews))
                print('Jerks: ', np.sum(jerks, axis=0).astype(int) / 1e4)
                print('constr_j_pos: ', np.sum(constrs['j_pos']))
                print('constr_j_vel: ', np.sum(constrs['j_vel']))
                print('constr_ee: ', np.sum(constrs['ee']))
                break


def test_env(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)
    env_info = env.env_info

    for i in range(iteration):
        rews = []
        jerks = []
        constrs = {'j_pos': [], 'j_vel': [], 'ee': []}

        obs = env.reset()
        env.render(mode="human")
        while True:
            act = env.action_space.sample()
            obs_, rew, done, info = env.step(act)
            env.render(mode='human')

            rews.append(rew)
            jerks.append(info['jerk'])
            constrs['j_pos'].append(np.any(info['constraints_value']['joint_pos_constr'] > 0).astype(int))
            constrs['j_vel'].append(np.any(info['constraints_value']['joint_vel_constr'] > 0).astype(int))
            constrs['ee'].append(np.any(info['constraints_value']['ee_constr'] > 0).astype(int))

            if done:
                print('Return: ', np.sum(rews))
                print('Jerks: ', np.sum(jerks, axis=0).astype(int) / 1e4)
                print('constr_j_pos: ', np.sum(constrs['j_pos']))
                print('constr_j_vel: ', np.sum(constrs['j_vel']))
                print('constr_ee: ', np.sum(constrs['ee']))
                break


def test_mp_env(env_id="3dof-hit-promp", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)

    for i in range(iteration):
        print("iteration: ", i)
        obs = env.reset()
        if i == 0:
            env.render(mode="human")
        while True:
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            if done:
                print('Return: ', np.sum(rew))
                print('Jerks: ', np.sum(info['jerk'], axis=0).astype(int) / 1e4)
                constrs = {'j_pos': [], 'j_vel': [], 'ee': []}
                for constr in info['constraints_value']:
                    constrs['j_pos'].append(np.any(constr['joint_pos_constr'] > 0).astype(int))
                    constrs['j_vel'].append(np.any(constr['joint_vel_constr'] > 0).astype(int))
                    constrs['ee'].append(np.any(constr['ee_constr'] > 0).astype(int))
                print('constr_j_pos: ', np.sum(constrs['j_pos']))
                print('constr_j_vel: ', np.sum(constrs['j_vel']))
                print('constr_ee: ', np.sum(constrs['ee']))
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


if __name__ == "__main__":
    # test_baseline(env_id='3dof-hit', iteration=10)
    # test_env(env_id="3dof-hit", iteration=10)
    test_mp_env(env_id="3dof-hit-promp", iteration=10)
    # test_mp()
