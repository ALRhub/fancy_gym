import os
from gym.envs.mujoco.hopper_v3 import HopperEnv

import numpy as np

MAX_EPISODE_STEPS_HOPPERTHROWINBASKET = 250


class ALRHopperThrowInBasketEnv(HopperEnv):
    """
    Initialization changes to normal Hopper:
    - healthy_reward: 1.0 -> 0.0
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - hit_basket_reward: - -> 10

    Reward changes to normal Hopper:
    - velocity: (x_position_after - x_position_before) -> (ball_position_after - ball_position_before)
    """

    def __init__(self,
                 xml_file='hopper_throw_in_basket.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=0.0,
                 hit_basket_reward=10,
                 basket_size=0.3,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-float('inf'), float('inf')),
                 reset_noise_scale=5e-3,
                 context=True,
                 penalty=0.0,
                 exclude_current_positions_from_observation=True,
                 max_episode_steps=250):
        self.hit_basket_reward = hit_basket_reward
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.ball_in_basket = False
        self.basket_size = basket_size
        self.context = context
        self.penalty = penalty
        self.basket_x = 5
        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
                         healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
                         exclude_current_positions_from_observation)

    def step(self, action):

        self.current_step += 1
        self.do_simulation(action, self.frame_skip)
        ball_pos = self.get_body_com("ball")

        basket_pos = self.get_body_com("basket_ground")
        basket_center = (basket_pos[0] + 0.5, basket_pos[1], basket_pos[2])

        is_in_basket_x = ball_pos[0] >= basket_pos[0] and ball_pos[0] <= basket_pos[0] + self.basket_size
        is_in_basket_y = ball_pos[1] >= basket_pos[1] - (self.basket_size / 2) and ball_pos[1] <= basket_pos[1] + (
                    self.basket_size / 2)
        is_in_basket_z = ball_pos[2] < 0.1
        is_in_basket = is_in_basket_x and is_in_basket_y and is_in_basket_z
        if is_in_basket: self.ball_in_basket = True

        ball_landed = self.get_body_com("ball")[2] <= 0.05
        done = ball_landed or is_in_basket

        rewards = 0

        ctrl_cost = self.control_cost(action)

        costs = ctrl_cost

        if self.current_step >= self.max_episode_steps or done:

            if is_in_basket:
                if not self.context:
                    rewards += self.hit_basket_reward
            else:
                dist = np.linalg.norm(ball_pos - basket_center)
                if self.context:
                    rewards = -10 * dist
                else:
                    rewards -= (dist * dist)
        else:
            # penalty not needed
            rewards += ((action[
                         :2] > 0) * self.penalty).sum() if self.current_step < 10 else 0  # too much of a penalty?

        observation = self._get_obs()
        reward = rewards - costs
        info = {
            'ball_pos': ball_pos[0],
        }

        return observation, reward, done, info

    def _get_obs(self):
        return np.append(super()._get_obs(), self.basket_x)

    def reset(self):
        if self.max_episode_steps == 10:
            # We have to initialize this here, because the spec is only added after creating the env.
            self.max_episode_steps = self.spec.max_episode_steps

        self.current_step = 0
        self.ball_in_basket = False
        if self.context:
            basket_id = self.sim.model.body_name2id("basket_ground")
            self.basket_x = self.np_random.uniform(3, 7, 1)
            self.sim.model.body_pos[basket_id] = [self.basket_x, 0, 0]
        return super().reset()

    # overwrite reset_model to make it deterministic
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation


if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    env = ALRHopperThrowInBasketEnv()
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
