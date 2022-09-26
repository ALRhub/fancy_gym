import os
from typing import Tuple, Union, Optional

import numpy as np
from gym.core import ObsType
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

MAX_EPISODE_STEPS_HALFCHEETAHJUMP = 100


class HalfCheetahJumpEnv(HalfCheetahEnv):
    """
    _ctrl_cost_weight 0.1 -> 0.0
    """

    def __init__(self,
                 xml_file='cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.0,
                 reset_noise_scale=0.1,
                 context=True,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=100):
        self.current_step = 0
        self.max_height = 0
        # self.max_episode_steps = max_episode_steps
        self.goal = 0
        self.context = context
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file=xml_file,
                         forward_reward_weight=forward_reward_weight,
                         ctrl_cost_weight=ctrl_cost_weight,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation)

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)

        height_after = self.get_body_com("torso")[2]
        self.max_height = max(height_after, self.max_height)

        ## Didnt use fell_over, because base env also has no done condition - Paul and Marc
        # fell_over = abs(self.sim.data.qpos[2]) > 2.5  # how to figure out if the cheetah fell over? -> 2.5 oke?
        # TODO: Should a fall over be checked here?
        done = False

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        if self.current_step == MAX_EPISODE_STEPS_HALFCHEETAHJUMP:
            height_goal_distance = -10 * np.linalg.norm(self.max_height - self.goal) + 1e-8 if self.context \
                else self.max_height
            rewards = self._forward_reward_weight * height_goal_distance
        else:
            rewards = 0

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'height': height_after,
            'max_height': self.max_height
        }

        return observation, reward, done, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.goal)

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None, ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.max_height = 0
        self.current_step = 0
        self.goal = self.np_random.uniform(1.1, 1.6, 1)  # 1.1 1.6
        return super().reset()

    # overwrite reset_model to make it deterministic
    def reset_model(self):
        # TODO remove if not needed!
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
