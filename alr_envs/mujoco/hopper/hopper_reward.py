import numpy as np
from alr_envs.mujoco import alr_reward_fct

class HopperReward(alr_reward_fct.AlrReward):
    def __init__(self, sim_steps):
        self.sim_steps = sim_steps

        self.height_traj = None
        self.max_height = None
        self.reset(None)

    def reset(self, context):
        self.height_traj = np.zeros(shape=(self.sim_steps, 1))
        self.max_height = 0
        self.context = None

    def compute_reward(self, fall_over, sim, step):
        height = sim.data.qpos[1]
        self.height_traj[step, :] = height
        if (height > self.max_height):
            self.max_height = height

        if step == self.sim_steps - 1 or fall_over:
            reward = self.max_height
            alive_bonus = 0.1 * step
            reward += alive_bonus
        else:
            # alive_bonus = 0.1 * step
            # reward = alive_bonus
            reward = 0

        return reward
