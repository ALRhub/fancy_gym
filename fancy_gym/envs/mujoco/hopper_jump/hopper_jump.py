import os

import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv, DEFAULT_CAMERA_CONFIG

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco

MAX_EPISODE_STEPS_HOPPERJUMP = 250


class HopperEnvCustomXML(HopperEnv):
    """
    Initialization changes to normal Hopper:
    - terminate_when_unhealthy: True -> False
    - healthy_reward: 1.0 -> 2.0
    - healthy_z_range: (0.7, float('inf')) -> (0.5, float('inf'))
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - exclude_current_positions_from_observation: True -> False
    """

    def __init__(
            self,
            xml_file,
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=(0.7, float("inf")),
            healthy_angle_range=(-0.2, 0.2),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=True,
            **kwargs,
    ):
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if not hasattr(self, 'observation_space'):
            if exclude_current_positions_from_observation:
                self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64
                )
            else:
                self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64
                )

        MujocoEnv.__init__(
            self,
            xml_file,
            4,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.render_active = False

    def render(self):
        self.render_active = True
        return super().render()


class HopperJumpEnv(HopperEnvCustomXML):
    """
    Initialization changes to normal Hopper:
    - terminate_when_unhealthy: True -> False
    - healthy_reward: 1.0 -> 2.0
    - healthy_z_range: (0.7, float('inf')) -> (0.5, float('inf'))
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - exclude_current_positions_from_observation: True -> False
    """

    def __init__(
            self,
            xml_file='hopper_jump.xml',
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=2.0,
            contact_weight=2.0,
            height_weight=10.0,
            dist_weight=3.0,
            terminate_when_unhealthy=False,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=(0.5, float('inf')),
            healthy_angle_range=(-float('inf'), float('inf')),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=False,
            sparse=False,
            **kwargs
    ):

        self.sparse = sparse
        self._height_weight = height_weight
        self._dist_weight = dist_weight
        self._contact_weight = contact_weight

        self.max_height = 0
        self.goal = np.zeros(3, )

        self._steps = 0
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.has_left_floor = False
        self.contact_dist = None

        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file=xml_file,
                         forward_reward_weight=forward_reward_weight,
                         ctrl_cost_weight=ctrl_cost_weight,
                         healthy_reward=healthy_reward,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_state_range=healthy_state_range,
                         healthy_z_range=healthy_z_range,
                         healthy_angle_range=healthy_angle_range,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                         **kwargs)

        # increase initial height
        self.init_qpos[1] = 1.5

    @property
    def exclude_current_positions_from_observation(self):
        return self._exclude_current_positions_from_observation

    def step(self, action):
        self._steps += 1

        self.do_simulation(action, self.frame_skip)

        height_after = self.get_body_com("torso")[2]
        # site_pos_after = self.data.get_site_xpos('foot_site')
        site_pos_after = self.data.site('foot_site').xpos
        self.max_height = max(height_after, self.max_height)

        has_floor_contact = self._is_floor_foot_contact() if not self.contact_with_floor else False

        if not self.init_floor_contact:
            self.init_floor_contact = has_floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not has_floor_contact
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = has_floor_contact

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        terminated = False
        truncated = False

        goal_dist = np.linalg.norm(site_pos_after - self.goal)
        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = goal_dist

        rewards = 0
        if not self.sparse or (self.sparse and self._steps >= MAX_EPISODE_STEPS_HOPPERJUMP):
            healthy_reward = self.healthy_reward
            distance_reward = -goal_dist * self._dist_weight
            height_reward = (self.max_height if self.sparse else height_after) * self._height_weight
            contact_reward = -(self.contact_dist or 5) * self._contact_weight
            rewards = self._forward_reward_weight * (distance_reward + height_reward + contact_reward + healthy_reward)

        observation = self._get_obs()
        reward = rewards - costs
        info = dict(
            height=height_after,
            x_pos=site_pos_after,
            max_height=self.max_height,
            goal=self.goal[:1],
            goal_dist=goal_dist,
            height_rew=self.max_height,
            healthy_reward=self.healthy_reward,
            healthy=self.is_healthy,
            contact_dist=self.contact_dist or 0
        )

        if self.render_active and self.render_mode=='human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # goal_dist = self.data.get_site_xpos('foot_site') - self.goal
        goal_dist = self.data.site('foot_site').xpos - self.goal
        return np.concatenate((super(HopperJumpEnv, self)._get_obs(), goal_dist.copy(), self.goal[:1]))

    def reset_model(self):
        # super(HopperJumpEnv, self).reset_model()

        # self.goal = self.np_random.uniform(0.3, 1.35, 1)[0]
        self.goal = np.concatenate([self.np_random.uniform(0.3, 1.35, 1), np.zeros(2, )])
        # self.sim.model.body_pos[self.sim.model.body_name2id('goal_site_body')] = self.goal
        self.model.body('goal_site_body').pos[:] = np.copy(self.goal)
        self.max_height = 0
        self._steps = 0

        noise_low = np.zeros(self.model.nq)
        noise_low[3] = -0.5
        noise_low[4] = -0.2

        noise_high = np.zeros(self.model.nq)
        noise_high[5] = 0.785

        qpos = (
            self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq) +
            self.init_qpos
        )
        qvel = (
            # self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv) +
            self.init_qvel
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.has_left_floor = False
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.contact_dist = None

        return observation

    def _is_floor_foot_contact(self):
        # floor_geom_id = self.model.geom_name2id('floor')
        # foot_geom_id = self.model.geom_name2id('foot_geom')
        # TODO: do this properly over a sensor in the xml file, see dmc hopper
        floor_geom_id = mujoco.mj_name2id(self.model,
                                          mujoco.mjtObj.mjOBJ_GEOM,
                                          'floor')
        foot_geom_id = mujoco.mj_name2id(self.model,
                                         mujoco.mjtObj.mjOBJ_GEOM,
                                         'foot_geom')
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            collision = contact.geom1 == floor_geom_id and contact.geom2 == foot_geom_id
            collision_trans = contact.geom1 == foot_geom_id and contact.geom2 == floor_geom_id
            if collision or collision_trans:
                return True
        return False

class HopperJumpMarkovRew(HopperJumpEnv):
    def step(self, action):
        self._steps += 1

        self.do_simulation(action, self.frame_skip)

        height_after = self.get_body_com("torso")[2]
        # site_pos_after = self.data.get_site_xpos('foot_site')
        site_pos_after = self.data.site('foot_site').xpos
        self.max_height = max(height_after, self.max_height)

        has_floor_contact = self._is_floor_foot_contact() if not self.contact_with_floor else False

        if not self.init_floor_contact:
            self.init_floor_contact = has_floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not has_floor_contact
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = has_floor_contact

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        terminated = False
        truncated = False

        goal_dist = np.linalg.norm(site_pos_after - self.goal)
        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = goal_dist

        rewards = 0
        if not self.sparse or (self.sparse and self._steps >= MAX_EPISODE_STEPS_HOPPERJUMP):
            healthy_reward = self.healthy_reward
            distance_reward = -goal_dist * self._dist_weight
            height_reward = (self.max_height if self.sparse else height_after) * self._height_weight
            contact_reward = -(self.contact_dist or 5) * self._contact_weight
            rewards = self._forward_reward_weight * (distance_reward + height_reward + contact_reward + healthy_reward)

        observation = self._get_obs()

        # While loop to simulate the process after jump to make the task Markovian
        if self.sparse and self.has_left_floor:
            while self._steps < MAX_EPISODE_STEPS_HOPPERJUMP:
                # Simulate to the end of the episode
                self._steps += 1

                try:
                    self.do_simulation(np.zeros_like(action), self.frame_skip)
                except Exception as e:
                    print(e)

                height_after = self.get_body_com("torso")[2]
                #site_pos_after = self.data.get_site_xpos('foot_site')
                site_pos_after = self.data.site('foot_site').xpos
                self.max_height = max(height_after, self.max_height)

                has_floor_contact = self._is_floor_foot_contact() if not self.contact_with_floor else False

                if not self.init_floor_contact:
                    self.init_floor_contact = has_floor_contact
                if self.init_floor_contact and not self.has_left_floor:
                    self.has_left_floor = not has_floor_contact
                if not self.contact_with_floor and self.has_left_floor:
                    self.contact_with_floor = has_floor_contact

                ctrl_cost = self.control_cost(action)
                costs = ctrl_cost
                done = False

                goal_dist = np.linalg.norm(site_pos_after - self.goal)
                if self.contact_dist is None and self.contact_with_floor:
                    self.contact_dist = goal_dist

                rewards = 0

            # Task has reached the end, compute the sparse reward
            done = True
            healthy_reward = self.healthy_reward
            distance_reward = -goal_dist * self._dist_weight
            height_reward = (self.max_height if self.sparse else height_after) * self._height_weight
            contact_reward = -(self.contact_dist or 5) * self._contact_weight
            rewards = self._forward_reward_weight * (distance_reward + height_reward + contact_reward + healthy_reward)

        reward = rewards - costs
        info = dict(
            height=height_after,
            x_pos=site_pos_after,
            max_height=self.max_height,
            goal=self.goal[:1],
            goal_dist=goal_dist,
            height_rew=self.max_height,
            healthy_reward=self.healthy_reward,
            healthy=self.is_healthy,
            contact_dist=self.contact_dist or 0,
            num_steps=self._steps,
            has_left_floor=self.has_left_floor
       )
        return observation, reward, terminated, truncated, info
