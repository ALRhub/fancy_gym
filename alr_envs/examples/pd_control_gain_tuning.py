import numpy as np
from matplotlib import pyplot as plt

from alr_envs import dmc
from alr_envs.utils.make_env_helpers import make_detpmp_env

# This might work for some environments, however, please verify either way the correct trajectory information
# for your environment are extracted below
SEED = 10
env_id = "cartpole-swingup"
wrappers = [dmc.suite.cartpole.MPWrapper]

mp_kwargs = {
    "num_dof": 1,
    "num_basis": 5,
    "duration": 2,
    "width": 0.025,
    "policy_type": "motor",
    "weights_scale": 0.2,
    "zero_start": True,
    "policy_kwargs": {
        "p_gains": 10,
        "d_gains": 10 # a good starting point is the sqrt of p_gains
    }
}

kwargs = dict(time_limit=2, episode_length=200)

env = make_detpmp_env(env_id, wrappers, seed=SEED, mp_kwargs=mp_kwargs,
                      **kwargs)

# Plot difference between real trajectory and target MP trajectory
env.reset()
pos, vel = env.mp_rollout(env.action_space.sample())

base_shape = env.full_action_space.shape
actual_pos = np.zeros((len(pos), *base_shape))
actual_pos_ball = np.zeros((len(pos), *base_shape))
actual_vel = np.zeros((len(pos), *base_shape))
act = np.zeros((len(pos), *base_shape))

for t, pos_vel in enumerate(zip(pos, vel)):
    actions = env.policy.get_action(pos_vel[0], pos_vel[1])
    actions = np.clip(actions, env.full_action_space.low, env.full_action_space.high)
    _, _, _, _ = env.env.step(actions)
    act[t, :] = actions
    # TODO verify for your environment
    actual_pos[t, :] = env.current_pos
    # actual_pos_ball[t, :] = env.physics.data.qpos[2:]
    actual_vel[t, :] = env.current_vel

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("Position")
plt.plot(actual_pos, c='C0', label=["true" if i == 0 else "" for i in range(np.prod(base_shape))])
# plt.plot(actual_pos_ball, label="true pos ball")
plt.plot(pos, c='C1', label=["MP" if i == 0 else "" for i in range(np.prod(base_shape))])
plt.xlabel("Episode steps")
plt.legend()

plt.subplot(132)
plt.title("Velocity")
plt.plot(actual_vel, c='C0', label=[f"true" if i == 0 else "" for i in range(np.prod(base_shape))])
plt.plot(vel, c='C1', label=[f"MP" if i == 0 else "" for i in range(np.prod(base_shape))])
plt.xlabel("Episode steps")
plt.legend()

plt.subplot(133)
plt.title("Actions")
plt.plot(act, c="C0"),  # label=[f"actions" if i == 0 else "" for i in range(np.prod(base_action_shape))])
plt.xlabel("Episode steps")
# plt.legend()
plt.show()
