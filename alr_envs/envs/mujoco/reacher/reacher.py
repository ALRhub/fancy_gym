import os

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv


class ReacherEnv(MujocoEnv, utils.EzPickle):
    """
    More general version of the gym mujoco Reacher environment
    """

    def __init__(self, sparse: bool = False, n_links: int = 5, ctrl_cost_weight: int = 1):
        utils.EzPickle.__init__(**locals())

        self._steps = 0
        self.n_links = n_links

        self.ctrl_cost_weight = ctrl_cost_weight

        self.sparse = sparse
        self.reward_weight = 1 if not sparse else 200

        file_name = f'reacher_{n_links}links.xml'

        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", file_name),
                           frame_skip=2,
                           mujoco_bindings="mujoco")

    def step(self, action):
        self._steps += 1

        is_reward = not self.sparse or (self.sparse and self._steps == 200)

        reward_dist = 0.0
        angular_vel = 0.0
        if is_reward:
            reward_dist = self.distance_reward()
            angular_vel = self.velocity_reward()

        reward_ctrl = -self.ctrl_cost_weight * np.square(action).sum()

        reward = reward_dist + reward_ctrl + angular_vel
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False

        infos = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            velocity=angular_vel,
            end_effector=self.get_body_com("fingertip").copy(),
            goal=self.goal if hasattr(self, "goal") else None
        )

        return ob, reward, done, infos

    def distance_reward(self):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        return -self.reward_weight * np.linalg.norm(vec)

    def velocity_reward(self):
        return -10 * np.square(self.data.qvel.flat[:self.n_links]).sum() if self.sparse else 0.0

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            # self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) +
            self.init_qpos.copy()
        )
        while True:
            # full space
            self.goal = self.np_random.uniform(low=-self.n_links / 10, high=self.n_links / 10, size=2)
            # I Quadrant
            # self.goal = self.np_random.uniform(low=0, high=self.n_links / 10, size=2)
            # II Quadrant
            # self.goal = np.random.uniform(low=[-self.n_links / 10, 0], high=[0, self.n_links / 10], size=2)
            # II + III Quadrant
            # self.goal = np.random.uniform(low=-self.n_links / 10, high=[0, self.n_links / 10], size=2)
            # I + II Quadrant
            # self.goal = np.random.uniform(low=[-self.n_links / 10, 0], high=self.n_links, size=2)
            if np.linalg.norm(self.goal) < self.n_links / 10:
                break

        qpos[-2:] = self.goal
        qvel = (
            # self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv) +
            self.init_qvel.copy()
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self._steps = 0

        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:self.n_links]
        target = self.get_body_com("target")
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            target[:2],  # x-y of goal position
            self.data.qvel.flat[:self.n_links],  # angular velocity
            self.get_body_com("fingertip") - target,  # goal distance
        ])
