import os
from typing import Optional, Any, Dict, Tuple

import numpy as np
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv, DEFAULT_CAMERA_CONFIG
from gymnasium.core import ObsType

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MAX_EPISODE_STEPS_WALKERJUMP = 300


# TODO: Right now this environment only considers jumping to a specific height, which is not nice. It should be extended
#  to the same structure as the Hopper, where the angles are randomized (->contexts) and the agent should jump as height
#  as possible, while landing at a specific target position

class Walker2dEnvCustomXML(Walker2dEnv):
    def __init__(
        self,
        xml_file,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
            )

        self.observation_space = observation_space

        MujocoEnv.__init__(
            self,
            xml_file,
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.render_active = False


class Walker2dJumpEnv(Walker2dEnvCustomXML):
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
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=300,
                 **kwargs):
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.max_height = 0
        self._penalty = penalty
        self.goal = 0
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file=xml_file,
                         forward_reward_weight=forward_reward_weight,
                         ctrl_cost_weight=ctrl_cost_weight,
                         healthy_reward=healthy_reward,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_z_range=healthy_z_range,
                         healthy_angle_range=healthy_angle_range,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                         **kwargs)

    def step(self, action):
        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        # pos_after = self.get_body_com("torso")[0]
        height = self.get_body_com("torso")[2]

        self.max_height = max(height, self.max_height)

        terminated = bool(height < 0.2)

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        rewards = 0
        if self.current_step >= self.max_episode_steps or terminated:
            terminated = True
            height_goal_distance = -10 * (np.linalg.norm(self.max_height - self.goal))
            healthy_reward = self.healthy_reward * self.current_step

            rewards = height_goal_distance + healthy_reward

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height,
            'max_height': self.max_height,
            'goal': self.goal,
        }
        truncated = False

        if self.render_active and self.render_mode=='human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        self.render_active = True
        return super().render()

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        self.current_step = 0
        self.max_height = 0
        ret = super().reset(seed=seed, options=options)
        self.goal = self.np_random.uniform(1.5, 2.5, 1)  # 1.5 3.0
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
