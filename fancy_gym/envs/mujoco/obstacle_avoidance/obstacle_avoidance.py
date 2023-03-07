import os
from typing import Optional

import mujoco
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_torque_max, q_min, q_max, get_quaternion_error, \
    rotation_distance

MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE = 250
GOAL_RANGE = np.array([0.2, 0.8])


class ObstacleAvoidanceEnv(MujocoEnv, utils.EzPickle):
    """
    More general version of the gym mujoco Reacher environment
    """

    def __init__(self, frame_skip=10):
        utils.EzPickle.__init__(**locals())

        self._steps = 0
        self.init_qpos_obs_avoidance = np.array([-3.56408685e-01, 4.29445454e-01, -1.35010595e-01, -2.05465189e+00,
                                                 9.35782229e-02, 2.48071794e+00, 2.32506759e-01, 1.00004915e-03,
                                                 9.99951374e-04])

        self.init_qvel_obs_avoidance = np.zeros(9)
        self.frame_skip = frame_skip
        self.goal_range = GOAL_RANGE
        self.obj_xy_list = []
        self._max_height = 0
        self._line_y_pos = 0
        self._max_height = 0
        self._desired_rod_quat = np.zeros(4)
        self.goal = np.zeros(2)
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "obstacle_avoidance.xml"),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self._line_y_pos = self.data.body('finish_line').xpos[1].copy()
        self.goal = self.data.site('target_pos').xpos[:2].copy()
        self._max_height = self.data.site('max_height').xpos[2].copy()
        self.obj_xy_list = [self.data.body('l1_obs').xpos[:2],
                            self.data.body('l2_top_obs').xpos[:2],
                            self.data.body('l2_bottom_obs').xpos[:2],
                            self.data.body('l3_top_obs').xpos[:2],
                            self.data.body('l3_mid_obs').xpos[:2],
                            self.data.body('l3_bottom_obs').xpos[:2]]
        self._desired_rod_quat = np.array([-7.80232724e-05, 9.99999177e-01, -1.15696870e-04, 1.27505693e-03])

    def step(self, action):

        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(action + self.data.qfrc_bias[:7].copy(), -q_torque_max, q_torque_max)

        unstable_simulation = False
        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1

        done = True if self._steps >= MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE else False

        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        if not unstable_simulation:
            reward, dist_to_obstacles_rew, dist_to_max_height_reward, ee_rot_rew, \
                action_reward = self._get_reward(rod_tip_pos, rod_quat, action)
        else:
            reward, dist_to_obstacles_rew, dist_to_max_height_reward, ee_rot_rew, \
                action_reward = -50, 0, 0, 0, 0

        ob = self._get_obs()
        infos = dict(
            tot_reward=reward,
            dist_to_obstacles_rew=dist_to_obstacles_rew,
            dist_to_max_height_reward=dist_to_max_height_reward,
            ee_rot_rew=ee_rot_rew,
            action_reward=action_reward
        )

        return ob, reward, done, infos

    def _get_reward(self, pos, rod_quat, action):
        def squared_exp_kernel(x, mean, scale, bandwidth):
            return scale * np.exp(
                np.square(np.linalg.norm(x - mean)) / bandwidth
            )

        rewards = 0
        # Distance to obstacles
        for obs in self.obj_xy_list:
            rewards -= squared_exp_kernel(pos[:2], np.array(obs), 1, 1)
        dist_to_obstacles_rew = np.copy(rewards)
        # rewards += np.abs(x[:, 1]- 0.4)
        dist_to_line_rew = -np.abs(pos[1] - self._line_y_pos)
        rewards += dist_to_line_rew
        # Distance to max height
        dist_to_max_height_reward = -squared_exp_kernel(pos[2], self._max_height, 10, 1)
        rewards += dist_to_max_height_reward
        # Correct ee rotation
        ee_rot_rew = 0
        rod_inclined_angle = rotation_distance(rod_quat, self._desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            ee_rot_rew -= 5*rod_inclined_angle / np.pi
        rewards += ee_rot_rew
        # action punishment
        action_reward = - 0.0005 * np.sum(np.square(action))
        rewards += action_reward
        return rewards, dist_to_obstacles_rew, dist_to_max_height_reward, ee_rot_rew, action_reward

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        if options is None or len(options.keys()) == 0:
            return super().reset()
        else:
            if self._mujoco_bindings.__name__ == "mujoco_py":
                self.sim.reset()
            else:
                self._mujoco_bindings.mj_resetData(self.model, self.data)
            return self.set_context(options['ctxt'])

    def set_context(self, context):
        self.reset_model()
        self.model.site('target_pos').pos = [context, self._line_y_pos, 0]
        self.goal = self.model.site('target_pos').pos.copy()[:2]
        return self._get_obs()

    def reset_model(self):
        self.set_state(self.init_qpos_obs_avoidance, self.init_qvel_obs_avoidance)
        pos = self.np_random.uniform(self.goal_range[0], self.goal_range[1])
        self.model.site('target_pos').pos = [pos, self._line_y_pos, 0]
        self.goal = self.model.site('target_pos').pos.copy()[:2]
        self._steps = 0
        return self._get_obs()

    def _get_obs(self):
        dist2goal = np.linalg.norm(self.data.site('target_pos').xpos.copy() - self.data.site("rod_tip").xpos.copy())
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            [self.goal[0]],  # goal x position
            [dist2goal],  # tip to goal distance
        ])
        return obs

    def sample_context(self):
        return self.np_random.uniform(self.goal_range[0], self.goal_range[1])

    def create_observation(self):
        return self._get_obs()

    def get_np_random(self):
        return self._np_random

    def calculateOfflineIK(self, desired_cart_pos, desired_cart_quat):
        """
        calculate offline inverse kinematics for franka pandas
        :param desired_cart_pos: desired cartesian position of tool center point
        :param desired_cart_quat: desired cartesian quaternion of tool center point
        :return: joint angles
        """
        J_reg = 1e-6
        w = np.diag([1, 1, 1, 1, 1, 1, 1])
        target_theta_null = np.array([
            3.57795216e-09,
            1.74532920e-01,
            3.30500960e-08,
            -8.72664630e-01,
            -1.14096181e-07,
            1.22173047e00,
            7.85398126e-01])
        eps = 1e-5  # threshold for convergence
        IT_MAX = 1000
        dt = 1e-3
        i = 0
        pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        pgain_null = 5 * np.array([
            7.675519770796831,
            2.676935478437176,
            8.539040163444975,
            1.270446361314313,
            8.87752182480855,
            2.186782233762969,
            4.414432577659688,
        ])
        pgain_limit = 20
        q = self.data.qpos[:7].copy()
        qd_d = np.zeros(q.shape)
        old_err_norm = np.inf

        while True:
            q_old = q
            q = q + dt * qd_d
            q = np.clip(q, q_min, q_max)
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            current_cart_pos = self.data.body("tcp").xpos.copy()
            current_cart_quat = self.data.body("tcp").xquat.copy()

            cart_pos_error = np.clip(desired_cart_pos - current_cart_pos, -0.1, 0.1)

            if np.linalg.norm(current_cart_quat - desired_cart_quat) > np.linalg.norm(
                    current_cart_quat + desired_cart_quat):
                current_cart_quat = -current_cart_quat
            cart_quat_error = np.clip(get_quaternion_error(current_cart_quat, desired_cart_quat), -0.5, 0.5)

            err = np.hstack((cart_pos_error, cart_quat_error))
            err_norm = np.sum(cart_pos_error ** 2) + np.sum((current_cart_quat - desired_cart_quat) ** 2)
            if err_norm > old_err_norm:
                q = q_old
                dt = 0.7 * dt
                continue
            else:
                dt = 1.025 * dt

            if err_norm < eps:
                break
            if i > IT_MAX:
                break

            old_err_norm = err_norm

            ### get Jacobian by mujoco
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)

            jacp = self.get_body_jacp("tcp")[:, :7].copy()
            jacr = self.get_body_jacr("tcp")[:, :7].copy()

            J = np.concatenate((jacp, jacr), axis=0)

            Jw = J.dot(w)

            # J * W * J.T + J_reg * I
            JwJ_reg = Jw.dot(J.T) + J_reg * np.eye(J.shape[0])

            # Null space velocity, points to home position
            qd_null = pgain_null * (target_theta_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (q_min + margin_to_limit - q)
            qd_null_limit[q > q_max - margin_to_limit] += qd_null_limit_max[q > q_max - margin_to_limit]
            qd_null_limit[q < q_min + margin_to_limit] += qd_null_limit_min[q < q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q

    def get_body_jacp(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, id)
        return jacp

    def get_body_jacr(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, id)
        return jacr


if __name__ == '__main__':
    env = ObstacleAvoidanceEnv()
    import time

    start_time = time.time()
    env.reset()
    # env.render()
    for _ in range(25000):
        # print(_)
        a = env.action_space.sample()
        obs, reward, done, infos = env.step(a)
        env.render()
        if done:
            env.reset()
            env.render()
    print("Test loop took: ", (time.time() - start_time) / 250)
