import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_trajs(start_index=0, end_index=150):
    # high resolution (1000 hz) traj
    dt_high = 0.001
    traj_high = np.load("./mp_traj_1000hz.npz", allow_pickle=True)
    pos_high = traj_high["position"]
    vel_high = traj_high["velocity"]
    acc_high = np.diff(vel_high, n=1, axis=0, append=np.zeros((1, 3))) / dt_high
    jer_high = np.diff(acc_high, n=1, axis=0, append=np.zeros((1, 3))) / dt_high
    time_high = np.linspace(0, 3.0, pos_high.shape[0])

    # low resolution (50 hz) traj
    dt_low = 0.02
    traj_low = np.load("./mp_traj_50hz.npz", allow_pickle=True)
    pos_low = traj_low["position"]
    vel_low = traj_low["velocity"]
    acc_low = np.diff(vel_low, n=1, axis=0, append=np.zeros((1, 3))) / dt_low
    jer_low = np.diff(acc_low, n=1, axis=0, append=np.zeros((1, 3))) / dt_low
    time_low = np.linspace(0, 3.0, pos_low.shape[0])

    # cubic polynomial interpolation traj
    tf = dt_low
    prev_pos = pos_low[0]
    prev_vel = vel_low[0]
    prev_acc = acc_low[0]
    pos_interp = [prev_pos]
    vel_interp = [prev_vel]
    acc_interp = [prev_acc]
    jer_interp = [np.zeros(3)]
    for i in range(pos_low.shape[0] - 1):
        coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
        results = np.vstack([prev_pos, pos_low[i+1], prev_vel, vel_low[i+1]])
        A = sp.linalg.block_diag(*[coef] * 3)
        y = results.reshape(-1, order='F')
        weights = np.linalg.solve(A, y).reshape(3, 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        jer_interp.append(np.abs(weights_dd[:, 1]) + np.abs(weights_dd[:, 0] - prev_acc) / dt_high)

        prev_pos = np.polynomial.polynomial.polyval(tf, weights.T)
        prev_vel = np.polynomial.polynomial.polyval(tf, weights_d.T)
        prev_acc = np.polynomial.polynomial.polyval(tf, weights_dd.T)

        for t in np.linspace(0.001, 0.02, 20):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            pos_interp.append(q)
            vel_interp.append(qd)
            acc_interp.append(qdd)

    pos_interp = np.array(pos_interp)
    vel_interp = np.array(vel_interp)
    acc_interp = np.array(acc_interp)
    jer_interp = np.array(jer_interp)

    constr_j_pos = [[-2.9, +2.9], [-1.8, +1.8], [-2.0, +2.0]]
    constr_j_vel = [[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]]
    constr_j_jerk = [[0, 1e4]] * 3

    s = start_index
    ss = 20 * s
    e = end_index
    ee = 20 * (e - 1) + 1
    for d in range(3):
        plt.subplot(3, 5, 5 * d + 1)
        if d == 0:
            plt.title("pos")
        plt.plot(time_high[ss:ee], pos_interp[ss:ee, d], color='red', label='interp_traj')
        plt.plot(time_high[ss:ee], pos_high[ss:ee, d], color='green', label='mp_1000hz')
        plt.plot(time_low[s:e], pos_low[s:e, d], color='blue', label='mp_50hz')
        plt.legend()
        # plt.hlines(constr_j_pos[d], xmin=time_low[0], xmax=time_low[-1], colors="r")

        plt.subplot(3, 5, 5 * d + 2)
        if d == 0:
            plt.title("vel")
        plt.plot(time_high[ss:ee], vel_interp[ss:ee, d], color='red')
        plt.plot(time_high[ss:ee], vel_high[ss:ee, d], color='green')
        plt.plot(time_low[s:e], vel_low[s:e, d], color='blue')
        # plt.hlines(constr_j_vel[d], xmin=time_low[0], xmax=time_low[-1], colors="r")

        plt.subplot(3, 5, 5 * d + 3)
        if d == 0:
            plt.title("acc")
        plt.plot(time_high[ss:ee], acc_interp[ss:ee, d], color='red')
        plt.plot(time_high[ss:ee], acc_high[ss:ee, d], color='green')
        plt.plot(time_low[s:e], acc_low[s:e, d], color='blue')

        plt.subplot(3, 5, 5 * d + 4)
        if d == 0:
            plt.title("mp jerk")
        plt.plot(time_high[ss:ee], jer_high[ss:ee, d], color='green')
        plt.plot(time_low[s:e], jer_low[s:e, d], color='blue')

        plt.subplot(3, 5, 5 * d + 5)
        if d == 0:
            plt.title("interp jerk")
        plt.plot(time_low[s:e], jer_interp[s:e, d], color='red')
        # plt.hlines(constr_j_jerk[d], xmin=time_low[0], xmax=time_low[-1], colors='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_trajs(start_index=0, end_index=150)
    plot_trajs(start_index=70, end_index=80)
