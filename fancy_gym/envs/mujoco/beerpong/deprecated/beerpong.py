import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

from fancy_gym.envs.mujoco.beerpong.deprecated.beerpong_reward_staged import BeerPongReward


class BeerPongEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=2):
        self._steps = 0
        # Small Context -> Easier. Todo: Should we do different versions?
        # self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
        #                              "beerpong_wo_cup" + ".xml")
        # self._cup_pos_min = np.array([-0.32, -2.2])
        # self._cup_pos_max = np.array([0.32, -1.2])

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets",
                                     "beerpong_wo_cup_big_table" + ".xml")
        self._cup_pos_min = np.array([-1.42, -4.05])
        self._cup_pos_max = np.array([1.42, -1.25])

        self._start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
        self._start_vel = np.zeros(7)

        self.release_step = 100  # time step of ball release
        self.ep_length = 600 // frame_skip

        self.reward_function = BeerPongReward()
        self.repeat_action = frame_skip
        self.model = None
        self.site_id = lambda x: self.model.site_name2id(x)
        self.body_id = lambda x: self.model.body_name2id(x)

        MujocoEnv.__init__(self, self.xml_path, frame_skip=1)
        utils.EzPickle.__init__(self)

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def start_vel(self):
        return self._start_vel

    def reset(self):
        self.reward_function.reset()
        return super().reset()

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        # TODO: Ask Max why we need to set the state twice.
        self.set_state(start_pos, init_vel)
        start_pos[7::] = self.sim.data.site_xpos[self.site_id("init_ball_pos"), :].copy()
        self.set_state(start_pos, init_vel)
        xy = self.np_random.uniform(self._cup_pos_min, self._cup_pos_max)
        xyz = np.zeros(3)
        xyz[:2] = xy
        xyz[-1] = 0.840
        self.sim.model.body_pos[self.body_id("cup_table")] = xyz
        return self._get_obs()

    def step(self, a):
        crash = False
        for _ in range(self.repeat_action):
            applied_action = a + self.sim.data.qfrc_bias[:len(a)].copy() / self.model.actuator_gear[:, 0]
            self.do_simulation(applied_action, self.frame_skip)
            self.reward_function.initialize(self)
            # self.reward_function.check_contacts(self.sim)   # I assume this is not important?
            if self._steps < self.release_step:
                self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.site_id("init_ball_pos"), :].copy()
                self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.site_id("init_ball_pos"), :].copy()
            crash = False

        ob = self._get_obs()

        if not crash:
            reward, reward_infos = self.reward_function.compute_reward(self, applied_action)
            is_collided = reward_infos['is_collided']
            terminated = is_collided or self._steps == self.ep_length - 1
            self._steps += 1
        else:
            reward = -30
            terminated = True
            reward_infos = {"success": False, "ball_pos": np.zeros(3), "ball_vel": np.zeros(3), "is_collided": False}

        infos = dict(
            reward=reward,
            action=a,
            q_pos=self.sim.data.qpos[0:7].ravel().copy(),
            q_vel=self.sim.data.qvel[0:7].ravel().copy(), sim_crash=crash,
        )
        infos.update(reward_infos)
        return ob, reward, terminated, infos

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        theta_dot = self.sim.data.qvel.flat[:7]
        ball_pos = self.data.get_body_xpos("ball").copy()
        cup_goal_diff_final = ball_pos - self.data.get_site_xpos("cup_goal_final_table").copy()
        cup_goal_diff_top = ball_pos - self.data.get_site_xpos("cup_goal_table").copy()
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            cup_goal_diff_final,
            cup_goal_diff_top,
            self.sim.model.body_pos[self.body_id("cup_table")][:2].copy(),
            [self._steps],
        ])

    @property
    def dt(self):
        return super(BeerPongEnv, self).dt * self.repeat_action


class BeerPongEnvFixedReleaseStep(BeerPongEnv):
    def __init__(self, frame_skip=2):
        super().__init__(frame_skip)
        self.release_step = 62  # empirically evaluated for frame_skip=2!


class BeerPongEnvStepBasedEpisodicReward(BeerPongEnv):
    def __init__(self, frame_skip=2):
        super().__init__(frame_skip)
        self.release_step = 62  # empirically evaluated for frame_skip=2!

    def step(self, a):
        if self._steps < self.release_step:
            return super(BeerPongEnvStepBasedEpisodicReward, self).step(a)
        else:
            reward = 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                sub_ob, sub_reward, terminated, truncated, sub_infos = super(BeerPongEnvStepBasedEpisodicReward,
                                                                             self).step(np.zeros(a.shape))
                reward += sub_reward
            infos = sub_infos
            ob = sub_ob
            ob[-1] = self.release_step + 1  # Since we simulate until the end of the episode, PPO does not see the
            # internal steps and thus, the observation also needs to be set correctly
        return ob, reward, terminated, truncated, infos


# class BeerBongEnvStepBased(BeerBongEnv):
#     def __init__(self, frame_skip=1, apply_gravity_comp=True, noisy=False, rndm_goal=False, cup_goal_pos=None):
#         super().__init__(frame_skip, apply_gravity_comp, noisy, rndm_goal, cup_goal_pos)
#         self.release_step = 62  # empirically evaluated for frame_skip=2!
#
#     def step(self, a):
#         if self._steps < self.release_step:
#             return super(BeerBongEnvStepBased, self).step(a)
#         else:
#             reward = 0
#             done = False
#             while not done:
#                 sub_ob, sub_reward, done, sub_infos = super(BeerBongEnvStepBased, self).step(np.zeros(a.shape))
#                 if not done or sub_infos['sim_crash']:
#                     reward += sub_reward
#                 else:
#                     ball_pos = self.sim.data.body_xpos[self.sim.model._body_name2id["ball"]].copy()
#                     cup_goal_dist_final = np.linalg.norm(ball_pos - self.sim.data.site_xpos[
#                         self.sim.model._site_name2id["cup_goal_final_table"]].copy())
#                     cup_goal_dist_top = np.linalg.norm(ball_pos - self.sim.data.site_xpos[
#                         self.sim.model._site_name2id["cup_goal_table"]].copy())
#                     if sub_infos['success']:
#                         dist_rew = -cup_goal_dist_final ** 2
#                     else:
#                         dist_rew = -0.5 * cup_goal_dist_final ** 2 - cup_goal_dist_top ** 2
#                     reward = reward - sub_infos['action_cost'] + dist_rew
#             infos = sub_infos
#             ob = sub_ob
#             ob[-1] = self.release_step + 1  # Since we simulate until the end of the episode, PPO does not see the
#             # internal steps and thus, the observation also needs to be set correctly
#         return ob, reward, done, infos
