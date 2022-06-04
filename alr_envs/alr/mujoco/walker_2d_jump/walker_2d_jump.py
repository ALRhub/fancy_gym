import os
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
import numpy as np

MAX_EPISODE_STEPS_WALKERJUMP = 300


class ALRWalker2dJumpEnv(Walker2dEnv):
    """
    healthy reward 1.0 -> 0.005 -> 0.0025 not from alex
    penalty 10 -> 0 not from alex
    """

    def __init__(self,
                 xml_file='walker2d.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.0025,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.8, 2.0),
                 healthy_angle_range=(-1.0, 1.0),
                 reset_noise_scale=5e-3,
                 penalty=0,
                 context=True,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=300):
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.max_height = 0
        self._penalty = penalty
        self.context = context
        self.goal = 0
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                         healthy_z_range, healthy_angle_range, reset_noise_scale,
                         exclude_current_positions_from_observation)
        
    def step(self, action):
        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        #pos_after = self.get_body_com("torso")[0]
        height = self.get_body_com("torso")[2]

        self.max_height = max(height, self.max_height)

        fell_over = height < 0.2
        done = fell_over

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        if self.current_step >= self.max_episode_steps or done:
            done = True
            height_goal_distance = -10 * (np.linalg.norm(self.max_height - self.goal)) if self.context else self.max_height
            healthy_reward = self.healthy_reward * self.current_step

            rewards = height_goal_distance + healthy_reward
        else:
            # penalty not needed
            rewards = 0
            rewards += ((action[:2] > 0) * self._penalty).sum() if self.current_step < 4 else 0
            rewards += ((action[3:5] > 0) * self._penalty).sum() if self.current_step < 4 else 0


        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height,
            'max_height': self.max_height,
            'goal' : self.goal,
        }

        return observation, reward, done, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self):
        self.current_step = 0
        self.max_height = 0
        self.goal = np.random.uniform(1.5, 2.5, 1) # 1.5 3.0
        return super().reset()

    # overwrite reset_model to make it deterministic
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    env = ALRWalker2dJumpEnv()
    obs = env.reset()

    for i in range(2000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        ac = env.action_space.sample()
        obs, rew, d, info = env.step(ac)
        if i % 10 == 0:
            env.render(mode=render_mode)
        if d:
            print('After ', i, ' steps, done: ', d)
            env.reset()

    env.close()