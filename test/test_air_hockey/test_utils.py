import numpy as np
import scipy
import matplotlib.pyplot as plt
import fancy_gym
from baseline.baseline_agent.baseline_agent import build_agent


def plot_trajs(position, velocity, start_index=0, end_index=100, plot_sampling=True, plot_constrs=True, dof=3):
    if plot_sampling:
        dt = 0.001
    else:
        dt = 0.02
    pos = position
    vel = velocity
    acc = np.diff(vel, n=1, axis=0, append=np.zeros([1, dof])) / dt
    jer = np.diff(acc, n=1, axis=0, append=np.zeros([1, dof])) / dt
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
        A = scipy.linalg.block_diag(*[coef] * dof)
        y = results.reshape(-1, order='F')
        weights = np.linalg.solve(A, y).reshape(dof, 4)
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

    if dof == 3:
        constr_j_pos = np.array([[-2.81, +2.81], [-1.70, +1.70], [-1.98, +1.98]])
        constr_j_vel = np.array([[-1.49, +1.49], [-1.49, +1.49], [-1.98, +1.98]])
        constr_j_jerk = [[0, 1e4]] * 3
    else:
        constr_j_pos = np.array([[-2.967, +2.967], [-2.094, +2.094], [-2.967, +2.967], [-2.094, +2.094],
                                 [-2.967, +2.967], [-2.094, +2.094], [-3.053, +3.054]])
        constr_j_vel = np.array([[-1.483, +1.483], [-1.483, +1.483], [-1.745, +1.745], [-1.308, +1.308],
                                 [-2.268, +2.268], [-2.356, +2.356], [-2.356, +2.356]])
        constr_j_jerk = [[0, 1e4]] * 7

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
    for d in range(dof):
        plt.subplot(dof, 4, 4 * d + 1)
        if d == 0:
            plt.title("mp_pos vs. interp_pos")
        plt.plot(step[s:e], pos[s:e, d], color='blue')
        plt.plot(interp_step[ss:ee], interp_pos[ss:ee, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_pos[d], xmin=step[s], xmax=step[e], colors="r")

        plt.subplot(dof, 4, 4 * d + 2)
        if d == 0:
            plt.title("mp_vel vs. interp_vel")
        plt.plot(step[s:e], vel[s:e, d], color='blue')
        plt.plot(interp_step[ss:ee], interp_vel[ss:ee, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_vel[d], xmin=step[s], xmax=step[e], colors="r")

        plt.subplot(dof, 4, 4 * d + 3)
        if d == 0:
            plt.title("mp_acc vs. interp_acc")
        plt.plot(interp_step[ss:ee], interp_acc[ss:ee, d], color='green')
        plt.plot(step[s:e], acc[s:e, d], color='blue')

        plt.subplot(dof, 4, 4 * d + 4)
        if d == 0:
            plt.title("mp_jerk vs. interp_jerk")
        plt.plot(step[s:e], jer[s:e, d], color='blue')
        plt.plot(step[s:e], interp_jer[s:e, d], color='green')
        if plot_constrs:
            plt.hlines(constr_j_jerk[d], xmin=step[s], xmax=step[e], colors='r')
    plt.show()


def plot_c_trajs():
    pass


def test_baseline(env_id="3dof-hit", seed=0, iteration=5):
    env_kwargs = {'interpolation_order': 3, 'custom_reward_function': 'HitSparseRewardV2',
                  'check_traj': False, 'check_step': False}
    env = fancy_gym.make(env_id=env_id, seed=seed, **env_kwargs)
    env_info = env.env_info
    agent = build_agent(env_info)

    for i in range(iteration):
        rews = []
        j_pos = []
        j_vel = []
        ee_pos = []
        ee_vel = []
        puck_pos = []
        puck_vel = []

        agent.reset()
        obs = env.reset()
        env.render(mode="human")
        while True:
            act = agent.draw_action(obs).reshape([-1])
            # act[:] = 0
            # act[0] = 0.5
            obs_, rew, done, info = env.step(act)
            env.render(mode="human")

            # print(info['compute_time_ms'])

            rews.append(rew)
            n_joints = env_info['robot']['n_joints']
            j_pos.append(act[:n_joints])
            j_vel.append(act[n_joints:])
            pos, vel = env.base_env.get_ee()
            ee_pos.append(pos)
            ee_vel.append(vel[-3:])
            pos, vel = env.base_env.get_puck(obs_)
            puck_pos.append(pos)
            puck_vel.append(vel)

            if done:
                print('Return: ', np.sum(rews))
                print('num_ee_x_violation: ', info['num_ee_x_violation'])
                print('num_ee_y_violation: ', info['num_ee_y_violation'])
                print('num_ee_z_violation: ', info['num_ee_z_violation'])
                print('num_jerk_violation: ', info['num_jerk_violation'])
                print('num_j_pos_violation: ', info['num_j_pos_violation'])
                print('num_j_vel_violation: ', info['num_j_vel_violation'])
                plot_trajs(np.array(j_pos), np.array(j_vel), plot_constrs=True, plot_sampling=False, dof=7)
                plot_trajs(np.array(ee_pos), np.array(ee_vel), plot_constrs=False, plot_sampling=False, dof=3)
                plot_trajs(np.array(puck_pos), np.array(puck_vel), plot_constrs=False, plot_sampling=False, dof=3)
                break


if __name__ == "__main__":
    test_baseline(env_id='7dof-hit', iteration=10)
