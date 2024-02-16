import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
from gymnasium.core import ObsType
from fancy_gym.envs.mujoco.hopper_jump.hopper_jump import HopperEnvCustomXML
from gymnasium import spaces


MAX_EPISODE_STEPS_HOPPERJUMPONBOX = 250


class HopperJumpOnBoxEnv(HopperEnvCustomXML):
    """
    Initialization changes to normal Hopper:
    - healthy_reward: 1.0 -> 0.01 -> 0.001
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    """

    def __init__(self,
                 xml_file='hopper_jump_on_box.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.001,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-float('inf'), float('inf')),
                 reset_noise_scale=5e-3,
                 context=True,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=250,
                 **kwargs):
        self.current_step = 0
        self.max_height = 0
        self.max_episode_steps = max_episode_steps
        self.min_distance = 5000  # what value?
        self.hopper_on_box = False
        self.context = context
        self.box_x = 1

        if exclude_current_positions_from_observation:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
            )

        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                         healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
                         exclude_current_positions_from_observation,
                         **kwargs)

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        height_after = self.get_body_com("torso")[2]
        foot_pos = self.get_body_com("foot")
        self.max_height = max(height_after, self.max_height)

        vx, vz, vangle = self.data.qvel[0:3]

        s = self.state_vector()
        fell_over = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height_after > .7))

        box_pos = self.get_body_com("box")
        box_size = 0.3
        box_height = 0.3
        box_center = (box_pos[0] + (box_size / 2), box_pos[1], box_height)
        foot_length = 0.3
        foot_center = foot_pos[0] - (foot_length / 2)

        dist = np.linalg.norm(foot_pos - box_center)

        self.min_distance = min(dist, self.min_distance)

        # check if foot is on box
        is_on_box_x = box_pos[0] <= foot_center <= box_pos[0] + box_size
        is_on_box_y = True  # is y always true because he can only move in x and z direction?
        is_on_box_z = box_height - 0.02 <= foot_pos[2] <= box_height + 0.02
        is_on_box = is_on_box_x and is_on_box_y and is_on_box_z
        if is_on_box:
            self.hopper_on_box = True

        ctrl_cost = self.control_cost(action)

        costs = ctrl_cost

        terminated = fell_over or self.hopper_on_box

        if self.current_step >= self.max_episode_steps or terminated:
            done = False  # TODO why are we doing this???

            max_height = self.max_height.copy()
            min_distance = self.min_distance.copy()

            alive_bonus = self._healthy_reward * self.current_step
            box_bonus = 0
            rewards = 0

            # TODO explain what we did here for the calculation of the reward
            if is_on_box:
                if self.context:
                    rewards -= 100 * vx ** 2 if 100 * vx ** 2 < 1 else 1
                else:
                    box_bonus = 10
                    rewards += box_bonus
                    # rewards -= dist * dist ???? why when already on box?
                    # reward -= 90 - abs(angle)
                    rewards -= 100 * vx ** 2 if 100 * vx ** 2 < 1 else 1
                    rewards += max_height * 3
                    rewards += alive_bonus

            else:
                if self.context:
                    rewards = -10 - min_distance
                    rewards += max_height * 3
                else:
                    # reward -= (dist*dist)
                    rewards -= min_distance * min_distance
                    # rewards -= dist / self.max_distance
                    rewards += max_height
                    rewards += alive_bonus

        else:
            rewards = 0

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height_after,
            'max_height': self.max_height.copy(),
            'min_distance': self.min_distance,
            'goal': self.box_x,
        }

        truncated = self.current_step >= self.max_episode_steps and not terminated

        if self.render_active and self.render_mode=='human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.box_x)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:

        self.max_height = 0
        self.min_distance = 5000
        self.current_step = 0
        self.hopper_on_box = False
        ret = super().reset(seed=seed, options=options)
        if self.context:
            self.box_x = self.np_random.uniform(1, 3, 1)
            self.model.body("box").pos = [self.box_x[0], 0, 0]
        return ret

    # overwrite reset_model to make it deterministic
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
