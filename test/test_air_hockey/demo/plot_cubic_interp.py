import numpy as np
import matplotlib.pyplot as plt
import scipy


def plot_cubic_interp(pos_list, vel_list):
    for i in range(len(pos_list)):
        pos = pos_list[i]
        vel = vel_list[i]

        # cubic spline
        tf = 0.02
        coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
        results = np.vstack([pos[0], pos[1], vel[0], vel[1]])

        A = scipy.linalg.block_diag(*[coef] * 3)
        y = results.reshape(-1, order='F')

        weights = np.linalg.solve(A, y).reshape(3, 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        interp_pos = []
        interp_vel = []
        interp_acc = []

        for t in np.linspace(0, 0.02, 21):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            interp_pos.append(q)
            interp_vel.append(qd)
            interp_acc.append(qdd)

        step = np.linspace(0.02, 0.04, 2)
        interp_step = np.linspace(0.02, 0.04, 21)

        interp_pos = np.array(interp_pos)
        interp_vel = np.array(interp_vel)
        interp_acc = np.array(interp_acc)

        for d in range(3):
            plt.subplot(3, 2, 2 * d + 1)
            plt.plot(step, pos[:, d], color='green')
            plt.plot(interp_step, interp_pos[:, d], color='blue')

            plt.subplot(3, 2, 2 * d + 2)
            plt.plot(step, vel[:, d], color='green')
            plt.plot(interp_step, interp_vel[:, d], color='blue')
    plt.show()


if __name__ == "__main__":
    # down sampling traj
    traj_high = np.load('./mp_traj_1000hz.npz', allow_pickle=True)
    pos_high = traj_high["position"]
    vel_high = traj_high["velocity"]
    steps = np.linspace(0, 3000, 151, dtype=np.int32)
    pos_down = pos_high[steps]
    vel_down = vel_high[steps]

    # low resolution traj
    traj_low = np.load("./mp_traj_50hz.npz", allow_pickle=True)
    pos_low = traj_low["position"]
    vel_low = traj_low["velocity"]

    position = [pos_down[60:62], pos_low[60:62]]
    velocity = [vel_down[60:62], vel_low[60:62]]

    plot_cubic_interp(position, velocity)

