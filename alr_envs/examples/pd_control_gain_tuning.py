import numpy as np
from matplotlib import pyplot as plt

from alr_envs import dmc, meta
from alr_envs.alr import mujoco
from alr_envs.utils.make_env_helpers import make_promp_env


def visualize(env):
    t = env.t
    pos_features = env.mp.basis_generator.basis(t)
    plt.plot(t, pos_features)
    plt.show()


# This might work for some environments, however, please verify either way the correct trajectory information
# for your environment are extracted below
SEED = 1
# env_id = "ball_in_cup-catch"
env_id = "ALRReacherSparse-v0"
env_id = "button-press-v2"
wrappers = [mujoco.reacher.MPWrapper]
wrappers = [meta.goal_object_change_mp_wrapper.MPWrapper]

mp_kwargs = {
    "num_dof": 4,
    "num_basis": 5,
    "duration": 6.25,
    "policy_type": "metaworld",
    "weights_scale": 10,
    "zero_start": True,
    # "policy_kwargs": {
    #     "p_gains": 1,
    #     "d_gains": 0.1
    # }
}

# kwargs = dict(time_limit=4, episode_length=200)
kwargs = {}

env = make_promp_env(env_id, wrappers, seed=SEED, mp_kwargs=mp_kwargs, **kwargs)
env.action_space.seed(SEED)

# Plot difference between real trajectory and target MP trajectory
env.reset()
w = env.action_space.sample()  # N(0,1)
visualize(env)
pos, vel = env.mp_rollout(w)

base_shape = env.full_action_space.shape
actual_pos = np.zeros((len(pos), *base_shape))
actual_vel = np.zeros((len(pos), *base_shape))
act = np.zeros((len(pos), *base_shape))

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
img = ax.imshow(env.env.render("rgb_array"))
fig.show()

for t, pos_vel in enumerate(zip(pos, vel)):
    actions = env.policy.get_action(pos_vel[0], pos_vel[1],, self.current_vel, self.current_pos
    actions = np.clip(actions, env.full_action_space.low, env.full_action_space.high)
    _, _, _, _ = env.env.step(actions)
    if t % 15 == 0:
        img.set_data(env.env.render("rgb_array"))
        fig.canvas.draw()
        fig.canvas.flush_events()
    act[t, :] = actions
    # TODO verify for your environment
    actual_pos[t, :] = env.current_pos
    actual_vel[t, :] = 0  # env.current_vel

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("Position")
p1 = plt.plot(actual_pos, c='C0', label="true")
# plt.plot(actual_pos_ball, label="true pos ball")
p2 = plt.plot(pos, c='C1', label="MP")  # , label=["MP" if i == 0 else None for i in range(np.prod(base_shape))])
plt.xlabel("Episode steps")
# plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
from collections import OrderedDict

by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.subplot(132)
plt.title("Velocity")
plt.plot(actual_vel, c='C0', label="true")
plt.plot(vel, c='C1', label="MP")
plt.xlabel("Episode steps")

plt.subplot(133)
plt.title(f"Actions {np.std(act, axis=0)}")
plt.plot(act, c="C0"),  # label=[f"actions" if i == 0 else "" for i in range(np.prod(base_action_shape))])
plt.xlabel("Episode steps")
# plt.legend()
plt.show()
