import os

import mujoco_py.builder
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv

from alr_envs.alr.mujoco.beerpong.beerpong_reward_staged import BeerPongReward

CUP_POS_MIN = np.array([-1.42, -4.05])
CUP_POS_MAX = np.array([1.42, -1.25])


# CUP_POS_MIN = np.array([-0.32, -2.2])
# CUP_POS_MAX = np.array([0.32, -1.2])

# smaller context space -> Easier task
# CUP_POS_MIN = np.array([-0.16, -2.2])
# CUP_POS_MAX = np.array([0.16, -1.7])

class ALRBeerBongEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=2, apply_gravity_comp=True):

        cup_goal_pos = np.array([-0.3, -1.2, 0.840])
        if cup_goal_pos.shape[0] == 2:
            cup_goal_pos = np.insert(cup_goal_pos, 2, 0.840)
        self.cup_goal_pos = np.array(cup_goal_pos)

        self._steps = 0
        # self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
        #                              "beerpong_wo_cup" + ".xml")
        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "beerpong_wo_cup_big_table" + ".xml")

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.apply_gravity_comp = apply_gravity_comp

        self._start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
        self._start_vel = np.zeros(7)

        self.ball_site_id = 0
        self.ball_id = 11

        self.release_step = 100  # time step of ball release

        self.ep_length = 600 // frame_skip
        self.cup_table_id = 10

        self.add_noise = False

        reward_function = BeerPongReward
        self.reward_function = reward_function()
        self.repeat_action = frame_skip
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

        self.set_state(start_pos, init_vel)
        start_pos[7::] = self.sim.data.site_xpos[self.ball_site_id, :].copy()
        self.set_state(start_pos, init_vel)
        xy = self.np_random.uniform(CUP_POS_MIN, CUP_POS_MAX)
        xyz = np.zeros(3)
        xyz[:2] = xy
        xyz[-1] = 0.840
        self.sim.model.body_pos[self.cup_table_id] = xyz
        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0

        for _ in range(self.repeat_action):
            if self.apply_gravity_comp:
                applied_action = a + self.sim.data.qfrc_bias[:len(a)].copy() / self.model.actuator_gear[:, 0]
            else:
                applied_action = a
            try:
                self.do_simulation(applied_action, self.frame_skip)
                self.reward_function.initialize(self)
                self.reward_function.check_contacts(self.sim)
                if self._steps < self.release_step:
                    self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.ball_site_id, :].copy()
                    self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.ball_site_id, :].copy()
                crash = False
            except mujoco_py.builder.MujocoException:
                crash = True

        ob = self._get_obs()

        if not crash:
            reward, reward_infos = self.reward_function.compute_reward(self, applied_action)
            success = reward_infos['success']
            is_collided = reward_infos['is_collided']
            ball_pos = reward_infos['ball_pos']
            ball_vel = reward_infos['ball_vel']
            done = is_collided or self._steps == self.ep_length - 1
            self._steps += 1
        else:
            reward = -30
            reward_infos = dict()
            success = False
            is_collided = False
            done = True
            ball_pos = np.zeros(3)
            ball_vel = np.zeros(3)

        infos = dict(
            reward_dist=reward_dist,
            reward=reward,
            velocity=angular_vel,
            # traj=self._q_pos,
            action=a,
            q_pos=self.sim.data.qpos[0:7].ravel().copy(),
            q_vel=self.sim.data.qvel[0:7].ravel().copy(),
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            success=success,
            is_collided=is_collided, sim_crash=crash,
            table_contact_first=int(not self.reward_function.ball_ground_contact_first)
        )
        infos.update(reward_infos)
        return ob, reward, done, infos

    def _check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        theta_dot = self.sim.data.qvel.flat[:7]
        ball_pos = self.sim.data.body_xpos[self.sim.model._body_name2id["ball"]].copy()
        cup_goal_diff_final = ball_pos - self.sim.data.site_xpos[
            self.sim.model._site_name2id["cup_goal_final_table"]].copy()
        cup_goal_diff_top = ball_pos - self.sim.data.site_xpos[self.sim.model._site_name2id["cup_goal_table"]].copy()
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            cup_goal_diff_final,
            cup_goal_diff_top,
            self.sim.model.body_pos[self.cup_table_id][:2].copy(),
            [self._steps],
        ])

    @property
    def dt(self):
        return super(ALRBeerBongEnv, self).dt * self.repeat_action


class ALRBeerBongEnvFixedReleaseStep(ALRBeerBongEnv):
    def __init__(self, frame_skip=2, apply_gravity_comp=True):
        super().__init__(frame_skip, apply_gravity_comp)
        self.release_step = 62  # empirically evaluated for frame_skip=2!


class ALRBeerBongEnvStepBasedEpisodicReward(ALRBeerBongEnv):
    def __init__(self, frame_skip=2, apply_gravity_comp=True):
        super().__init__(frame_skip, apply_gravity_comp)
        self.release_step = 62  # empirically evaluated for frame_skip=2!

    def step(self, a):
        if self._steps < self.release_step:
            return super(ALRBeerBongEnvStepBasedEpisodicReward, self).step(a)
        else:
            reward = 0
            done = False
            while not done:
                sub_ob, sub_reward, done, sub_infos = super(ALRBeerBongEnvStepBasedEpisodicReward, self).step(
                    np.zeros(a.shape))
                reward += sub_reward
            infos = sub_infos
            ob = sub_ob
            ob[-1] = self.release_step + 1  # Since we simulate until the end of the episode, PPO does not see the
            # internal steps and thus, the observation also needs to be set correctly
        return ob, reward, done, infos


# class ALRBeerBongEnvStepBased(ALRBeerBongEnv):
#     def __init__(self, frame_skip=1, apply_gravity_comp=True, noisy=False, rndm_goal=False, cup_goal_pos=None):
#         super().__init__(frame_skip, apply_gravity_comp, noisy, rndm_goal, cup_goal_pos)
#         self.release_step = 62  # empirically evaluated for frame_skip=2!
#
#     def step(self, a):
#         if self._steps < self.release_step:
#             return super(ALRBeerBongEnvStepBased, self).step(a)
#         else:
#             reward = 0
#             done = False
#             while not done:
#                 sub_ob, sub_reward, done, sub_infos = super(ALRBeerBongEnvStepBased, self).step(np.zeros(a.shape))
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


if __name__ == "__main__":
    env = ALRBeerBongEnv(frame_skip=2)
    # env = ALRBeerBongEnvStepBased(frame_skip=2)
    # env = ALRBeerBongEnvStepBasedEpisodicReward(frame_skip=2)
    # env = ALRBeerBongEnvFixedReleaseStep(frame_skip=2)
    import time

    env.reset()
    env.render("human")
    for i in range(1500):
        # ac = 10 * env.action_space.sample()
        ac = np.ones(7)
        # ac = np.zeros(7)
        # ac[0] = 0
        # ac[1] = -0.01
        # ac[3] = -0.01
        # if env._steps > 150:
        #     ac[0] = 1
        obs, rew, d, info = env.step(ac)
        env.render("human")
        print(env.dt)
        print(rew)

        if d:
            print('RESETTING')
            env.reset()
            time.sleep(1)
    env.close()
