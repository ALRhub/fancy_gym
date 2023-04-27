import numpy as np
import torch as tc
import matplotlib.pyplot as plt

from fancy_gym.black_box.factory.phase_generator_factory import get_phase_generator
from fancy_gym.black_box.factory.basis_generator_factory import get_basis_generator
from fancy_gym.black_box.factory.trajectory_generator_factory import get_trajectory_generator


def test_mp():
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
    pass
