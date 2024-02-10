from typing import Tuple, Union, Optional, Any, Dict

import numpy as np
from gymnasium.core import ObsType
from gymnasium.envs.mujoco.ant_v4 import AntEnv, DEFAULT_CAMERA_CONFIG
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MAX_EPISODE_STEPS_ANTJUMP = 200


# TODO: This environment was not tested yet. Do the following todos and test it.
# TODO: Right now this environment only considers jumping to a specific height, which is not nice. It should be extended
#  to the same structure as the Hopper, where the angles are randomized (->contexts) and the agent should jump as heigh
#  as possible, while landing at a specific target position

class AntEnvCustomXML(AntEnv):
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 27 + 1
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


class AntJumpEnv(AntEnvCustomXML):
    """
    Initialization changes to normal Ant:
    - healthy_reward: 1.0 -> 0.01 -> 0.0 no healthy reward needed - Paul and Marc
    - _ctrl_cost_weight 0.5 -> 0.0
    - contact_cost_weight: 5e-4 -> 0.0
    - healthy_z_range: (0.2, 1.0) -> (0.3, float('inf'))  !!!!! Does that make sense, limiting height?
    """

    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.0,
                 contact_cost_weight=0.0,
                 healthy_reward=0.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.3, float('inf')),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 **kwargs
                 ):
        self.current_step = 0
        self.max_height = 0
        self.goal = 0
        super().__init__(xml_file=xml_file,
                         ctrl_cost_weight=ctrl_cost_weight,
                         contact_cost_weight=contact_cost_weight,
                         healthy_reward=healthy_reward,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_z_range=healthy_z_range,
                         contact_force_range=contact_force_range,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation, **kwargs)
        self.render_active = False

    def step(self, action):
        self.current_step += 1
        self.do_simulation(action, self.frame_skip)

        height = self.get_body_com("torso")[2].copy()

        self.max_height = max(height, self.max_height)

        rewards = 0

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        costs = ctrl_cost + contact_cost

        terminated = bool(
            height < 0.3)  # fall over -> is the 0.3 value from healthy_z_range? TODO change 0.3 to the value of healthy z angle

        if self.current_step == MAX_EPISODE_STEPS_ANTJUMP or terminated:
            # -10 for scaling the value of the distance between the max_height and the goal height; only used when context is enabled
            # height_reward = -10 * (np.linalg.norm(self.max_height - self.goal))
            height_reward = -10 * np.linalg.norm(self.max_height - self.goal)
            # no healthy reward when using context, because we optimize a negative value
            healthy_reward = 0

            rewards = height_reward + healthy_reward

        obs = self._get_obs()
        reward = rewards - costs

        info = {
            'height': height,
            'max_height': self.max_height,
            'goal': self.goal
        }
        truncated = False

        if self.render_active and self.render_mode=='human':
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.render_active = True
        return super().render()

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        self.current_step = 0
        self.max_height = 0
        # goal heights from 1.0 to 2.5; can be increased, but didnt work well with CMORE
        ret = super().reset(seed=seed, options=options)
        self.goal = self.np_random.uniform(1.0, 2.5, 1)
        return ret

    # reset_model had to be implemented in every env to make it deterministic
    def reset_model(self):
        # Todo remove if not needed
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
