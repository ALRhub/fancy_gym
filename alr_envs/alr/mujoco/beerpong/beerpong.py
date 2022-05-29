import mujoco_py.builder
import os

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
    def __init__(self, frame_skip=1, apply_gravity_comp=True, noisy=False,
                 rndm_goal=False, cup_goal_pos=None):
        cup_goal_pos = np.array(cup_goal_pos if cup_goal_pos is not None else [-0.3, -1.2, 0.840])
        if cup_goal_pos.shape[0]==2:
            cup_goal_pos = np.insert(cup_goal_pos, 2, 0.840)
        self.cup_goal_pos = np.array(cup_goal_pos)

        self._steps = 0
        # self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
        #                              "beerpong_wo_cup" + ".xml")
        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "beerpong_wo_cup_big_table" + ".xml")

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.rndm_goal = rndm_goal
        self.apply_gravity_comp = apply_gravity_comp
        self.add_noise = noisy

        self._start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
        self._start_vel = np.zeros(7)

        self.ball_site_id = 0
        self.ball_id = 11

        # self._release_step = 175  # time step of ball release
        # self._release_step = 130  # time step of ball release
        self.release_step = 100  # time step of ball release

        self.ep_length = 600  # based on 3 seconds with dt = 0.005 int(self.sim_time / self.dt)
        self.cup_table_id = 10

        if noisy:
            self.noise_std = 0.01
        else:
            self.noise_std = 0
        reward_function = BeerPongReward
        self.reward_function = reward_function()

        MujocoEnv.__init__(self, self.xml_path, frame_skip)
        utils.EzPickle.__init__(self)

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def start_vel(self):
        return self._start_vel

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def reset(self):
        self.reward_function.reset(self.add_noise)
        return super().reset()

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        self.set_state(start_pos, init_vel)
        self.sim.model.body_pos[self.cup_table_id] = self.cup_goal_pos
        start_pos[7::] = self.sim.data.site_xpos[self.ball_site_id, :].copy()
        self.set_state(start_pos, init_vel)
        if self.rndm_goal:
            xy = self.np_random.uniform(CUP_POS_MIN, CUP_POS_MAX)
            xyz = np.zeros(3)
            xyz[:2] = xy
            xyz[-1] = 0.840
            self.sim.model.body_pos[self.cup_table_id] = xyz
        return self._get_obs()

    def step(self, a):
        # if a.shape[0] == 8: # we learn also when to release the ball
        #     self._release_step = a[-1]
        # self._release_step = np.clip(self._release_step, 50, 250)
        # self.release_step = 0.5/self.dt
        reward_dist = 0.0
        angular_vel = 0.0
        applied_action = a
        reward_ctrl = - np.square(applied_action).sum()
        if self.apply_gravity_comp:
            applied_action += self.sim.data.qfrc_bias[:len(applied_action)].copy() / self.model.actuator_gear[:, 0]
        try:
            self.do_simulation(applied_action, self.frame_skip)
            if self._steps < self.release_step:
                self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.ball_site_id, :].copy()
                self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.ball_site_id, :].copy()
            elif self._steps == self.release_step and self.add_noise:
                 self.sim.data.qvel[7::] += self.noise_std * np.random.randn(3)
            crash = False
        except mujoco_py.builder.MujocoException:
            crash = True
        # joint_cons_viol = self.check_traj_in_joint_limits()

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

        infos = dict(reward_dist=reward_dist,
                     reward_ctrl=reward_ctrl,
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
                     table_contact_first=int(not self.reward_function.ball_ground_contact_first))
        infos.update(reward_infos)

        return ob, reward, done, infos

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.model.body_pos[self.cup_table_id][:2].copy(),
            [self._steps],
        ])

    # TODO
    @property
    def active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [True] * 2,  # xy position of cup
            [False]  # env steps
        ])


if __name__ == "__main__":
    env = ALRBeerBongEnv(rndm_goal=True)
    import time
    env.reset()
    env.render("human")
    for i in range(1500):
        # ac = 10 * env.action_space.sample()[0:7]
        ac = np.zeros(7)
        obs, rew, d, info = env.step(ac)
        env.render("human")

        print(rew)

        if d:
            print('RESETTING')
            env.reset()
            time.sleep(1)
    env.close()
