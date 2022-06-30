from gym.envs.mujoco.hopper_v3 import HopperEnv
import numpy as np
import os

MAX_EPISODE_STEPS_HOPPERJUMP = 250


class ALRHopperJumpEnv(HopperEnv):
    """
    Initialization changes to normal Hopper:
    - healthy_reward: 1.0 -> 0.1 -> 0
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - healthy_z_range: (0.7, float('inf')) -> (0.5, float('inf'))
    - exclude current positions from observatiosn is set to False
    """

    def __init__(
            self,
            xml_file='hopper_jump.xml',
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=0.0,
            penalty=0.0,
            context=True,
            terminate_when_unhealthy=False,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=(0.5, float('inf')),
            healthy_angle_range=(-float('inf'), float('inf')),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=False,
            max_episode_steps=250
            ):

        self.current_step = 0
        self.max_height = 0
        self.max_episode_steps = max_episode_steps
        self.penalty = penalty
        self.goal = 0
        self.context = context
        self.exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._floor_geom_id = None
        self._foot_geom_id = None
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.has_left_floor = False
        self.contact_dist = None
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                         healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
                         exclude_current_positions_from_observation)

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        # site_pos_after = self.sim.data.site_xpos[self.model.site_name2id('foot_site')].copy()
        site_pos_after = self.get_body_com('foot_site')
        self.max_height = max(height_after, self.max_height)

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = False
        rewards = 0
        if self.current_step >= self.max_episode_steps:
            hight_goal_distance = -10 * np.linalg.norm(self.max_height - self.goal) if self.context else self.max_height
            healthy_reward = 0 if self.context else self.healthy_reward * 2  # self.current_step
            height_reward = self._forward_reward_weight * hight_goal_distance  # maybe move reward calculation into if structure and define two different _forward_reward_weight variables for context and episodic seperatley
            rewards = height_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - costs

        info = {
            'height': height_after,
            'x_pos': site_pos_after,
            'max_height': self.max_height,
            'height_rew': self.max_height,
            'healthy_reward': self.healthy_reward * 2,
            'healthy': self.is_healthy
            }

        return observation, reward, done, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self):
        self.goal = self.np_random.uniform(1.4, 2.16, 1)[0]  # 1.3 2.3
        self.max_height = 0
        self.current_step = 0
        return super().reset()

    # overwrite reset_model to make it deterministic
    def reset_model(self):

        qpos = self.init_qpos  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.has_left_floor = False
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.contact_dist = None
        return observation

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            collision = con.geom1 == id_1 and con.geom2 == id_2
            collision_trans = con.geom1 == id_2 and con.geom2 == id_1
            if collision or collision_trans:
                return True
        return False


class ALRHopperXYJumpEnv(ALRHopperJumpEnv):

    def step(self, action):
        self._floor_geom_id = self.model.geom_name2id('floor')
        self._foot_geom_id = self.model.geom_name2id('foot_geom')

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        site_pos_after = self.sim.data.site_xpos[self.model.site_name2id('foot_site')].copy()
        self.max_height = max(height_after, self.max_height)

        # floor_contact = self._contact_checker(self._floor_geom_id, self._foot_geom_id) if not self.contact_with_floor else False
        # self.init_floor_contact = floor_contact if not self.init_floor_contact else self.init_floor_contact
        # self.has_left_floor = not floor_contact if self.init_floor_contact and not self.has_left_floor else self.has_left_floor
        # self.contact_with_floor = floor_contact if not self.contact_with_floor and self.has_left_floor else self.contact_with_floor

        floor_contact = self._contact_checker(self._floor_geom_id,
                                              self._foot_geom_id) if not self.contact_with_floor else False
        if not self.init_floor_contact:
            self.init_floor_contact = floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not floor_contact
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = floor_contact

        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = np.linalg.norm(self.sim.data.site_xpos[self.model.site_name2id('foot_site')]
                                               - np.array([self.goal, 0, 0]))

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = False
        goal_dist = np.linalg.norm(site_pos_after - np.array([self.goal, 0, 0]))
        rewards = 0
        if self.current_step >= self.max_episode_steps:
            # healthy_reward = 0 if self.context else self.healthy_reward * self.current_step
            healthy_reward = self.healthy_reward * 2  # * self.current_step
            contact_dist = self.contact_dist if self.contact_dist is not None else 5
            dist_reward = self._forward_reward_weight * (-3 * goal_dist + 10 * self.max_height - 2 * contact_dist)
            rewards = dist_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height_after,
            'x_pos': site_pos_after,
            'max_height': self.max_height,
            'goal': self.goal,
            'goal_dist': goal_dist,
            'height_rew': self.max_height,
            'healthy_reward': self.healthy_reward * 2,
            'healthy': self.is_healthy,
            'contact_dist': self.contact_dist if self.contact_dist is not None else 0
            }
        return observation, reward, done, info

    def reset_model(self):
        self.init_qpos[1] = 1.5
        self._floor_geom_id = self.model.geom_name2id('floor')
        self._foot_geom_id = self.model.geom_name2id('foot_geom')
        noise_low = -np.zeros(self.model.nq)
        noise_low[3] = -0.5
        noise_low[4] = -0.2
        noise_low[5] = 0

        noise_high = np.zeros(self.model.nq)
        noise_high[3] = 0
        noise_high[4] = 0
        noise_high[5] = 0.785

        rnd_vec = self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qpos = self.init_qpos + rnd_vec
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.has_left_floor = False
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.contact_dist = None

        return observation

    def reset(self):
        super().reset()
        self.goal = self.np_random.uniform(0.3, 1.35, 1)[0]
        self.sim.model.body_pos[self.sim.model.body_name2id('goal_site_body')] = np.array([self.goal, 0, 0])
        return self.reset_model()

    def _get_obs(self):
        goal_diff = self.sim.data.site_xpos[self.model.site_name2id('foot_site')].copy() \
                    - np.array([self.goal, 0, 0])
        return np.concatenate((super(ALRHopperXYJumpEnv, self)._get_obs(), goal_diff))

    def set_context(self, context):
        # context is 4 dimensional
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[-3:] = context[:3]
        self.goal = context[-1]
        self.set_state(qpos, qvel)
        self.sim.model.body_pos[self.sim.model.body_name2id('goal_site_body')] = np.array([self.goal, 0, 0])
        return self._get_obs()


class ALRHopperXYJumpEnvStepBased(ALRHopperXYJumpEnv):

    def __init__(
            self,
            xml_file='hopper_jump.xml',
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=0.0,
            penalty=0.0,
            context=True,
            terminate_when_unhealthy=False,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=(0.5, float('inf')),
            healthy_angle_range=(-float('inf'), float('inf')),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=False,
            max_episode_steps=250,
            height_scale=10,
            dist_scale=3,
            healthy_scale=2
            ):
        self.height_scale = height_scale
        self.dist_scale = dist_scale
        self.healthy_scale = healthy_scale
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, penalty, context,
                         terminate_when_unhealthy, healthy_state_range, healthy_z_range, healthy_angle_range,
                         reset_noise_scale, exclude_current_positions_from_observation, max_episode_steps)

    def step(self, action):
        self._floor_geom_id = self.model.geom_name2id('floor')
        self._foot_geom_id = self.model.geom_name2id('foot_geom')

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        site_pos_after = self.sim.data.site_xpos[self.model.site_name2id('foot_site')].copy()
        self.max_height = max(height_after, self.max_height)
        ctrl_cost = self.control_cost(action)

        healthy_reward = self.healthy_reward * self.healthy_scale
        height_reward = self.height_scale * height_after
        goal_dist = np.atleast_1d(np.linalg.norm(site_pos_after - np.array([self.goal, 0, 0], dtype=object)))[0]
        goal_dist_reward = -self.dist_scale * goal_dist
        dist_reward = self._forward_reward_weight * (goal_dist_reward + height_reward)
        reward = -ctrl_cost + healthy_reward + dist_reward
        done = False
        observation = self._get_obs()

        ###########################################################
        # This is only for logging the distance to goal when first having the contact
        ##########################################################
        floor_contact = self._contact_checker(self._floor_geom_id,
                                              self._foot_geom_id) if not self.contact_with_floor else False
        if not self.init_floor_contact:
            self.init_floor_contact = floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not floor_contact
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = floor_contact

        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = np.linalg.norm(self.sim.data.site_xpos[self.model.site_name2id('foot_site')]
                                               - np.array([self.goal, 0, 0]))
        info = {
            'height': height_after,
            'x_pos': site_pos_after,
            'max_height': self.max_height,
            'goal': self.goal,
            'goal_dist': goal_dist,
            'height_rew': self.max_height,
            'healthy_reward': self.healthy_reward * self.healthy_reward,
            'healthy': self.is_healthy,
            'contact_dist': self.contact_dist if self.contact_dist is not None else 0
            }
        return observation, reward, done, info


if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    # env = ALRHopperJumpEnv()
    # env = ALRHopperXYJumpEnv()
    np.random.seed(0)
    env = ALRHopperXYJumpEnvStepBased()
    env.seed(0)
    # env = ALRHopperJumpRndmPosEnv()
    obs = env.reset()

    for k in range(1000):
        obs = env.reset()
        print('observation :', obs[:])
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
