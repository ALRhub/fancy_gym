import mujoco_py.builder
import os

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv


class ALRBeerBongEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=1, apply_gravity_comp=True, reward_type: str = "staged", noisy=False,
                 context: np.ndarray = None, difficulty='simple'):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "beerpong_wo_cup" + ".xml")

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.context = context
        self.apply_gravity_comp = apply_gravity_comp
        self.add_noise = noisy

        self._start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
        self._start_vel = np.zeros(7)

        self.ball_site_id = 0
        self.ball_id = 11

        self._release_step = 175  # time step of ball release

        self.sim_time = 3  # seconds
        self.ep_length = 600  # based on 3 seconds with dt = 0.005 int(self.sim_time / self.dt)
        self.cup_table_id = 10

        if noisy:
            self.noise_std = 0.02
        else:
            self.noise_std = 0

        if difficulty == 'simple':
            self.cup_goal_pos = np.array([0, -1.7, 0.840])
        elif difficulty == 'intermediate':
            self.cup_goal_pos = np.array([0.3, -1.5, 0.840])
        elif difficulty == 'hard':
            self.cup_goal_pos = np.array([-0.3, -2.2, 0.840])
        elif difficulty == 'hardest':
            self.cup_goal_pos = np.array([-0.3, -1.2, 0.840])

        if reward_type == "no_context":
            from alr_envs.alr.mujoco.beerpong.beerpong_reward import BeerPongReward
            reward_function = BeerPongReward
        elif reward_type == "staged":
            from alr_envs.alr.mujoco.beerpong.beerpong_reward_staged import BeerPongReward
            reward_function = BeerPongReward
        else:
            raise ValueError("Unknown reward type: {}".format(reward_type))
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
        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0
        reward_ctrl = - np.square(a).sum()

        if self.apply_gravity_comp:
            a = a + self.sim.data.qfrc_bias[:len(a)].copy() / self.model.actuator_gear[:, 0]
        try:
            self.do_simulation(a, self.frame_skip)
            if self._steps < self._release_step:
                self.sim.data.qpos[7::] = self.sim.data.site_xpos[self.ball_site_id, :].copy()
                self.sim.data.qvel[7::] = self.sim.data.site_xvelp[self.ball_site_id, :].copy()
            elif self._steps == self._release_step and self.add_noise:
                 self.sim.data.qvel[7::] += self.noise_std * self.np_random.randn(3)
            crash = False
        except mujoco_py.builder.MujocoException:
            crash = True
        # joint_cons_viol = self.check_traj_in_joint_limits()

        ob = self._get_obs()

        if not crash:
            reward, reward_infos = self.reward_function.compute_reward(self, a)
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
                     is_collided=is_collided, sim_crash=crash)
        infos.update(reward_infos)

        return ob, reward, done, infos

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    # TODO: extend observation space
    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])

    # TODO
    @property
    def active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            # [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])


if __name__ == "__main__":
    env = ALRBeerBongEnv(reward_type="staged", difficulty='hardest')

    # env.configure(ctxt)
    env.reset()
    env.render("human")
    for i in range(800):
        ac = env.action_space.sample()[0:7]
        obs, rew, d, info = env.step(ac)
        env.render("human")

        print(rew)

        if d:
            break

    env.close()
