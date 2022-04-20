from gym.envs.mujoco.hopper_v3 import HopperEnv
import numpy as np

MAX_EPISODE_STEPS_HOPPERJUMP = 250


class ALRHopperJumpEnv(HopperEnv):
    """
    Initialization changes to normal Hopper:
    - healthy_reward: 1.0 -> 0.1 -> 0
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - healthy_z_range: (0.7, float('inf')) -> (0.5, float('inf'))

    """

    def __init__(self,
                 xml_file='hopper.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.0,
                 penalty=0.0,
                 context=True,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.5, float('inf')),
                 healthy_angle_range=(-float('inf'), float('inf')),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=250):
        self.current_step = 0
        self.max_height = 0
        self.max_episode_steps = max_episode_steps
        self.penalty = penalty
        self.goal = 0
        self.context = context
        self.exclude_current_positions_from_observation = exclude_current_positions_from_observation
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                         healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
                         exclude_current_positions_from_observation)

    def step(self, action):
        
        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        self.max_height = max(height_after, self.max_height)

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = False

        if self.current_step >= self.max_episode_steps:
            hight_goal_distance = -10*np.linalg.norm(self.max_height - self.goal) if self.context else self.max_height
            healthy_reward = 0 if self.context else self.healthy_reward * self.current_step
            height_reward = self._forward_reward_weight * hight_goal_distance # maybe move reward calculation into if structure and define two different _forward_reward_weight variables for context and episodic seperatley
            rewards = height_reward + healthy_reward

        else:
            # penalty for wrong start direction of first two joints; not needed, could be removed
            rewards = ((action[:2] > 0) * self.penalty).sum() if self.current_step < 10 else 0

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height'    : height_after,
            'max_height': self.max_height,
            'goal' : self.goal
        }

        return observation, reward, done, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self):
        self.goal = np.random.uniform(1.4, 2.3, 1) # 1.3 2.3
        self.max_height = 0
        self.current_step = 0
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


class ALRHopperJumpRndmPosEnv(ALRHopperJumpEnv):
    def __init__(self, max_episode_steps=250):
        super(ALRHopperJumpRndmPosEnv, self).__init__(exclude_current_positions_from_observation=False,
                                                      reset_noise_scale=5e-1,
                                                      max_episode_steps=max_episode_steps)
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel #+ self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        self.max_height = max(height_after, self.max_height)

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = False

        if self.current_step >= self.max_episode_steps:
            healthy_reward = 0
            height_reward = self._forward_reward_weight * self.max_height  # maybe move reward calculation into if structure and define two different _forward_reward_weight variables for context and episodic seperatley
            rewards = height_reward + healthy_reward

        else:
            # penalty for wrong start direction of first two joints; not needed, could be removed
            rewards = ((action[:2] > 0) * self.penalty).sum() if self.current_step < 10 else 0

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height_after,
            'max_height': self.max_height,
            'goal': self.goal
        }

        return observation, reward, done, info

if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    env = ALRHopperJumpEnv()
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