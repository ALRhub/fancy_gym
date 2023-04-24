import numpy as np
import matplotlib.pyplot as plt
import scipy


def plot_trajs():
    tf = 0.02
    pos = np.array([[-0.9650644, +0.8820466, +0.9497525], [-0.95893276, +0.8893953, +0.9668013]])
    vel = np.array([[+0.3065887, +0.3674361, +0.8524366], [+0.30082494, +0.3671441, +0.8609094]])
    vel = np.array([[+0.3082975, +0.3685026, +0.8482952], [+0.30420791, +0.3667898, +0.8569555]])
    acc = np.array([[-0.2881884, -0.0146031, +0.4236400], [-0.50619394, +0.1338124, +0.3653764]])
    acc = np.array([[-0.2044841, -0.0856384, +0.4330158], [-0.39635301, +0.0625819, +0.4023134]])
    # err = (pos[0] + vel[1] * tf) - pos[0]

    ##
    coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
    # coef = np.array([[1, 0, 0, 0, 0, 0], [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
    #                  [0, 1, 0, 0, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
    #                  [0, 0, 2, 0, 0, 0], [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])
    results = np.vstack([pos[0], pos[1], vel[0], vel[1]])
    # results = np.vstack([pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]])
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
        plt.subplot(3, 3, 3 * d + 1)
        plt.plot(step, pos[:, d])
        plt.plot(interp_step, interp_pos[:, d])

        plt.subplot(3, 3, 3 * d + 2)
        plt.plot(step, vel[:, d])
        plt.plot(interp_step, interp_vel[:, d])

        plt.subplot(3, 3, 3 * d + 3)
        plt.plot(step, acc[:, d])
        plt.plot(interp_step, interp_acc[:, d])
    plt.show()


if __name__ == "__main__":
    plot_trajs()

