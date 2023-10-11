from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

import gymnasium as gym
import fancy_gym

# This might work for some environments, however, please verify either way the correct trajectory information
# for your environment are extracted below
SEED = 1

env_id = "fancy_ProMP/Reacher5d-v0"

env = fancy_gym.make(env_id, mp_config_override={'controller_kwargs': {'p_gains': 0.05, 'd_gains': 0.05}}).env
env.action_space.seed(SEED)

# Plot difference between real trajectory and target MP trajectory
env.reset(seed=SEED)
w = env.action_space.sample()
pos, vel = env.get_trajectory(w)

base_shape = env.env.action_space.shape
actual_pos = np.zeros((len(pos), *base_shape))
actual_vel = np.zeros((len(pos), *base_shape))
act = np.zeros((len(pos), *base_shape))

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

img = ax.imshow(env.env.render(mode="rgb_array"))
fig.show()

for t, (des_pos, des_vel) in enumerate(zip(pos, vel)):
    actions = env.tracking_controller.get_action(des_pos, des_vel, env.current_pos, env.current_vel)
    actions = np.clip(actions, env.env.action_space.low, env.env.action_space.high)
    env.env.step(actions)
    if t % 15 == 0:
        img.set_data(env.env.render(mode="rgb_array"))
        fig.canvas.draw()
        fig.canvas.flush_events()
    act[t, :] = actions
    # TODO verify for your environment
    actual_pos[t, :] = env.current_pos
    actual_vel[t, :] = env.current_vel

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("Position")
p1 = plt.plot(actual_pos, c='C0', label="true")
p2 = plt.plot(pos, c='C1', label="MP")
plt.xlabel("Episode steps")
handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.subplot(132)
plt.title("Velocity")
plt.plot(actual_vel, c='C0', label="true")
plt.plot(vel, c='C1', label="MP")
plt.xlabel("Episode steps")

plt.subplot(133)
plt.title(f"Actions {np.std(act, axis=0)}")
plt.plot(act, c="C0"),
plt.xlabel("Episode steps")
plt.show()
