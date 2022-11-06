import fancy_gym
import numpy as np
import matplotlib.pyplot as plt

# This is the code that I am using to plot the data


def plot_trajs(desired_traj, actual_traj, dim):
    fig, ax = plt.subplots()
    ax.plot(desired_traj[:, dim], label='desired')
    ax.plot(actual_traj[:, dim], label='actual')
    ax.legend()
    plt.show()


def compare_desired_and_actual(env_id: str = "TableTennis4DProMP-v0"):
    env = fancy_gym.make(env_id, seed=0)
    env.traj_gen.basis_gn.show_basis(plot=True)
    env.reset()
    for _ in range(1):
        env.render(mode=None)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        for i in range(1):
            plot_trajs(info['desired_pos_traj'], info['pos_traj'], i)
            # plot_trajs(info['desired_vel_traj'], info['vel_traj'], i)
        if done:
            env.reset()

if __name__ == "__main__":
    compare_desired_and_actual(env_id='TableTennis4DProMP-v0')