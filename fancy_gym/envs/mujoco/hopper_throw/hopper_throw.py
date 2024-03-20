import os
from typing import Optional, Any, Dict, Tuple

import numpy as np
from gymnasium.core import ObsType
from fancy_gym.envs.mujoco.hopper_jump.hopper_jump import HopperEnvCustomXML
from gymnasium import spaces

MAX_EPISODE_STEPS_HOPPERTHROW = 250


class HopperThrowEnv(HopperEnvCustomXML):
    """
    Initialization changes to normal Hopper:
    - healthy_reward: 1.0 -> 0.0 -> 0.1
    - forward_reward_weight -> 5.0
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))

    Reward changes to normal Hopper:
    - velocity: (x_position_after - x_position_before) -> self.get_body_com("ball")[0]
    """

    def __init__(self,
                 xml_file='hopper_throw.xml',
                 forward_reward_weight=5.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.1,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-float('inf'), float('inf')),
                 reset_noise_scale=5e-3,
                 context=True,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=250,
                 **kwargs):
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.context = context
        self.goal = 0

        if not hasattr(self, 'observation_space'):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
            )

        super().__init__(xml_file=xml_file,
                         forward_reward_weight=forward_reward_weight,
                         ctrl_cost_weight=ctrl_cost_weight,
                         healthy_reward=healthy_reward,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_angle_range=healthy_state_range,
                         healthy_z_range=healthy_z_range,
                         healthy_state_range=healthy_angle_range,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                         **kwargs)

        self.render_active = False

    def step(self, action):
        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        ball_pos_after = self.get_body_com("ball")[
            0]  # abs(self.get_body_com("ball")[0]) # use x and y to get point and use euclid distance as reward?
        ball_pos_after_y = self.get_body_com("ball")[2]

        # done = self.done TODO We should use this, not sure why there is no other termination; ball_landed should be enough, because we only look at the throw itself? - Paul and Marc
        ball_landed = bool(self.get_body_com("ball")[2] <= 0.05)
        terminated = ball_landed

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        rewards = 0

        if self.current_step >= self.max_episode_steps or terminated:
            distance_reward = -np.linalg.norm(ball_pos_after - self.goal) if self.context else \
                self._forward_reward_weight * ball_pos_after
            healthy_reward = 0 if self.context else self.healthy_reward * self.current_step

            rewards = distance_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'ball_pos': ball_pos_after,
            'ball_pos_y': ball_pos_after_y,
            '_steps': self.current_step,
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
        ret = super().reset(seed=seed, options=options)
        self.goal = self.goal = self.np_random.uniform(2.0, 6.0, 1)  # 0.5 8.0
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
