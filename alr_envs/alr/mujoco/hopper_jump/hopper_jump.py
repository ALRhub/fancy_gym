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

        qpos = self.init_qpos # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation


class ALRHopperJumpRndmPosEnv(ALRHopperJumpEnv):
    def __init__(self, max_episode_steps=250):
        self.contact_with_floor = False
        self._floor_geom_id = None
        self._foot_geom_id = None
        super(ALRHopperJumpRndmPosEnv, self).__init__(exclude_current_positions_from_observation=False,
                                                      reset_noise_scale=5e-1,
                                                      max_episode_steps=max_episode_steps)

    def reset_model(self):
        self._floor_geom_id = self.model.geom_name2id('floor')
        self._foot_geom_id = self.model.geom_name2id('foot_geom')
        noise_low = -np.ones(self.model.nq)*self._reset_noise_scale
        noise_low[1] = 0
        noise_low[2] = -0.3
        noise_low[3] = -0.1
        noise_low[4] = -1.1
        noise_low[5] = -0.785

        noise_high = np.ones(self.model.nq)*self._reset_noise_scale
        noise_high[1] = 0
        noise_high[2] = 0.3
        noise_high[3] = 0
        noise_high[4] = 0
        noise_high[5] = 0.785

        rnd_vec = self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        # rnd_vec[2] *= 0.05  # the angle around the y axis shouldn't be too high as the agent then falls down quickly and
                            # can not recover
        # rnd_vec[1] = np.clip(rnd_vec[1], 0, 0.3)
        qpos = self.init_qpos + rnd_vec
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)

        self.contact_with_floor = self._contact_checker(self._floor_geom_id, self._foot_geom_id) if not \
                                                                      self.contact_with_floor else True

        height_after = self.get_body_com("torso")[2]
        self.max_height = max(height_after, self.max_height) if self.contact_with_floor else 0

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

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            collision = con.geom1 == id_1 and con.geom2 == id_2
            collision_trans = con.geom1 == id_2 and con.geom2 == id_1
            if collision or collision_trans:
                return True
        return False

if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    # env = ALRHopperJumpEnv()
    env = ALRHopperJumpRndmPosEnv()
    obs = env.reset()

    for k in range(10):
        obs = env.reset()
        print('observation :', obs[:6])
        for i in range(200):
            # objective.load_result("/tmp/cma")
            # test with random actions
            ac = env.action_space.sample()
            obs, rew, d, info = env.step(ac)
            # if i % 10 == 0:
            #     env.render(mode=render_mode)
            env.render(mode=render_mode)
            if d:
                print('After ', i, ' steps, done: ', d)
                env.reset()

    env.close()