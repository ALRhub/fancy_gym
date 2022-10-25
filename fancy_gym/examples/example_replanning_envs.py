import fancy_gym
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(traj):
    plt.figure()
    plt.plot(traj[:, 3])
    plt.legend()
    plt.show()

def run_replanning_envs(env_name="BoxPushingProDMP-v0", seed=1, iterations=1, render=True):
    env = fancy_gym.make(env_name, seed=seed)
    env.reset()
    for i in range(iterations):
        done = False
        desired_pos_traj = np.zeros((100, 7))
        desired_vel_traj = np.zeros((100, 7))
        real_pos_traj = np.zeros((100, 7))
        real_vel_traj = np.zeros((100, 7))
        t = 0
        while done is False:
            ac = env.action_space.sample()
            obs, reward, done, info = env.step(ac)
            desired_pos_traj[t: t + 25, :] = info['desired_pos']
            desired_vel_traj[t: t + 25, :] = info['desired_vel']
            # real_pos_traj.append(info['current_pos'])
            # real_vel_traj.append(info['current_vel'])
            t += 25
            if render:
                env.render(mode="human")
            if done:
                env.reset()
        plot_trajectory(desired_pos_traj)
    env.close()
    del env

if __name__ == "__main__":
    run_replanning_envs(env_name="BoxPushingDenseProDMP-v0", seed=1, iterations=1, render=False)