import os
from typing import Optional

import mujoco
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_torque_max, q_min, q_max, get_quaternion_error, \
    rotation_distance

MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE = 100
# GOAL_RANGE = np.array([0.2, 0.8])
GOAL_RANGE = np.array([0.4, 0.6])
TASK_SPACE_MIN = np.array([0.2, -0.25])
TASK_SPACE_MAX = np.array([0.7, 0.5])


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
        self._desired_rod_quat = np.zeros(4)
        self.goal = np.zeros(2)
        self.init_z = 0
        self.desired_positions = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.desired_positions_after_clip = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.actual_positions = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.reward_traj = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE + 1, 1))
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "obstacle_avoidance.xml"),
                           frame_skip=frame_skip,
                           mujoco_bindings="mujoco")
        # self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.action_space_cart = spaces.Box(low=-10, high=10, shape=(2,))
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,))
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
        unstable_simulation = False
        des_pos = None
        for _ in range(self.frame_skip):
            if action.shape[0] != 7:
                if des_pos is None:
                    # des_pos = self.data.body("rod_tip").xpos[:2].copy() + np.clip(action, self.action_space_cart.low,
                    #                                                               self.action_space_cart.high)
                    des_pos = np.clip(action, self.action_space_cart.low, self.action_space_cart.high)
                    self.desired_positions[self._steps] = des_pos
                    des_pos = np.clip(des_pos, TASK_SPACE_MIN, TASK_SPACE_MAX)
                    self.desired_positions_after_clip[self._steps] = des_pos
                    self.actual_positions[self._steps] = self.data.body("rod_tip").xpos[:2].copy()
                    des_pos = np.concatenate([des_pos, [self.init_z]])
                # print(des_pos)
                torques = self.map2torque(des_pos, np.array([0, 1, 0, 0]))
            else:
                torques = action

            # torques = 10 * np.clip(torques, self.action_space.low, self.action_space.high)
            torques = 10 * np.clip(torques, -1, 1)
            resultant_action = np.clip(torques + self.data.qfrc_bias[:7].copy(), -q_torque_max, q_torque_max)
            # print(resultant_action)
            try:
                # self.do_simulation(resultant_action, self.frame_skip)
                # self.do_simulation(resultant_action, 1)  # do simulation will check if the action space shape aligns
                # with the action ... this won't work as we control in torque
                # but action space are xy positions
                self.data.ctrl[:] = resultant_action
                self._mujoco_bindings.mj_step(self.model, self.data)
            except Exception as e:
                print(e)
                unstable_simulation = True
                break

        self._steps += 1

        done = True if self._steps >= MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE else False

        rod_tip_pos = self.data.body("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        if not unstable_simulation:
            reward, dist_to_obstacles_rew, dist_to_goal_rew = self._get_reward(rod_tip_pos)
        else:
            reward, dist_to_obstacles_rew, dist_to_goal_rew = -50, 0, 0
        self.reward_traj[self._steps, 0] = reward
        ob = self._get_obs()
        infos = dict(
            tot_reward=reward,
            dist_to_obstacles_rew=dist_to_obstacles_rew,
            dist_to_goal_rew=dist_to_goal_rew,
            distance_to_goal=np.linalg.norm(self.goal[:2] - self.data.body("rod_tip").xpos[:2].copy())
        )

        return ob, reward, done, infos

    def _get_reward(self, pos):
        def squared_exp_kernel(x, mean, scale, bandwidth):
            return scale * np.exp(np.square(np.linalg.norm(x - mean)) / bandwidth)

        def quad(x, goal, scale):
            return scale * np.linalg.norm(x - goal) ** 2

        rewards = 0
        # Distance to obstacles
        for obs in self.obj_xy_list:
            rewards -= squared_exp_kernel(pos[:2], np.array(obs), 5.0, 1)
        dist_to_obstacles_rew = np.copy(rewards)
        # dist_to_obstacles_rew = 0

        # rewards += np.abs(x[:, 1]- 0.4)

        additional_dist = 10 * np.linalg.norm(pos[:2] - self.goal)
        rewards -= additional_dist

        dist_to_line_rew = -np.abs(pos[1] - self._line_y_pos)

        rewards += dist_to_line_rew
        dist_to_goal_rew = -squared_exp_kernel(pos[:2], self.goal, 10, 1)
        # dist_to_goal_rew = -quad(pos[:2], self.goal, 100)
        rewards += dist_to_goal_rew
        return rewards, dist_to_obstacles_rew, dist_to_goal_rew

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
        self.model.body('finish_line').pos[1] = 0.45
        self._line_y_pos = self.model.body('finish_line').pos[1]
        pos = self.np_random.uniform(self.goal_range[0], self.goal_range[1])
        # pos = 0.35
        self.model.site('target_pos').pos = [pos, self._line_y_pos, 0]
        self.goal = self.model.site('target_pos').pos.copy()[:2]
        self._steps = 0
        self.init_z = self.data.body("rod_tip").xpos[-1].copy()
        self.actual_positions = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.desired_positions = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.desired_positions_after_clip = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE, 2))
        self.reward_traj = np.zeros((MAX_EPISODE_STEPS_OBSTACLEAVOIDANCE + 1, 1))
        return self._get_obs()

    def _get_obs(self):
        dist2goal = np.linalg.norm(self.data.site('target_pos').xpos.copy() - self.data.body("rod_tip").xpos.copy())
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

    def map2torque(self, desired_cart_pos, desired_cart_quat):
        pgain_pos = np.array([200.0, 200.0, 800.0])
        pgain_quat = np.array([30.0, 30.0, 30.0])

        pgain_joint = np.array([120.0, 120.0, 120.0, 120.0, 50.0, 30.0, 10.0])
        dgain_joint = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 5.0, 3.0])

        w = np.diag([1, 1, 1, 1, 1, 1, 1])
        J_reg = 1e-12
        min_svd_values = 1e-2
        max_svd_values = 1e2
        rest_posture = np.array([0, 0.174, 0, -0.872, 0, 1.222, 0.785])
        pgain_null = np.array([40, 40, 40, 40, 40, 40, 40])
        joint_pos_max = np.array([2.8973, 1.7628, 2.0, -0.0698, 2.8973, 3.7525, 2.8973])
        joint_pos_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        learning_rate = 0.001
        dt = 0.002
        ddgain = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

        old_state = self.data.qpos.copy()
        old_velocities = self.data.qvel.copy()
        q = self.data.qpos[:7].copy()

        joint_filter_coefficient = 1
        q = (joint_filter_coefficient * q + (1 - joint_filter_coefficient) * q)

        qd_dsum = np.zeros(q.shape)

        des_quat = desired_cart_quat
        for i in range(3):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            current_c_pos = self.data.body("rod_tip").xpos.copy()
            current_c_quat = self.data.body("rod_tip").xquat.copy()
            target_cpos_acc = desired_cart_pos - current_c_pos
            curr_quat = current_c_quat

            if np.linalg.norm(curr_quat - des_quat) > np.linalg.norm(curr_quat + des_quat):
                des_quat = -des_quat

            target_cquat = get_quaternion_error(curr_quat, des_quat)
            target_cpos_acc = np.clip(target_cpos_acc, -0.01, 0.01)
            target_cquat = np.clip(target_cquat, -0.1, 0.1)

            target_c_acc = np.hstack((pgain_pos * target_cpos_acc, pgain_quat * target_cquat))

            jacp = self.get_body_jacp("rod_tip")[:, :7].copy()
            jacr = self.get_body_jacr("rod_tip")[:, :7].copy()
            J = np.concatenate((jacp, jacr), axis=0)

            # Singular Value decomposition, to clip the singular values which are too small/big

            Jw = J.dot(w)

            # J *  W * J' + reg * I
            JwJ_reg = Jw.dot(J.T) + J_reg * np.eye(J.shape[0])

            u, s, v = np.linalg.svd(JwJ_reg, full_matrices=False)
            s = np.clip(s, min_svd_values, max_svd_values)
            # reconstruct the Jacobian
            JwJ_reg = u @ np.diag(s) @ v

            qdev_rest = np.clip(rest_posture - q, -0.2, 0.2)

            # Null space movement
            qd_null = np.array(pgain_null * qdev_rest)

            margin_to_limit = 0.01
            pgain_limit = 20

            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (joint_pos_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (joint_pos_min + margin_to_limit - q)
            qd_null_limit[q > joint_pos_max - margin_to_limit] += qd_null_limit_max[q > joint_pos_max - margin_to_limit]
            qd_null_limit[q < joint_pos_min + margin_to_limit] += qd_null_limit_min[q < joint_pos_min + margin_to_limit]

            # qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            # clip desired joint velocities for stability

            if np.linalg.norm(qd_d) > 3:
                qd_d = qd_d * 3 / np.linalg.norm(qd_d)

            qd_dsum = qd_dsum + qd_d

            q = q + learning_rate * qd_d
            q = np.clip(q, joint_pos_min, joint_pos_max)

        qd_dsum = (q - old_state[:7]) / dt
        # des_acc = ddgain * (qd_dsum - np.zeros(7)) / dt  # is the np.zeros(7) fine?

        qd_d = q - old_state[:7]
        vd_d = qd_dsum - old_velocities[:7]

        target_j_acc = pgain_joint * qd_d + dgain_joint * vd_d  # original
        return target_j_acc

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

    def plot_trajs(self):
        import matplotlib.pyplot as plt
        plt.figure()
        for k in range(self.desired_positions_after_clip.shape[1]):
            plt.subplot(self.desired_positions_after_clip.shape[1] + 1, 1, k + 1)
            plt.plot(self.desired_positions_after_clip[:, k], 'red')
            plt.plot(self.desired_positions[:, k], '--', color='red')
            plt.plot(self.actual_positions[:, k], 'blue')
            plt.plot(self.actual_positions[:, k].shape[0], self.goal[k], 'x')
        plt.subplot(self.desired_positions_after_clip.shape[1] + 1, 1, self.desired_positions_after_clip.shape[1] + 1)
        plt.plot(self.reward_traj)
        plt.show()


if __name__ == '__main__':
    env = ObstacleAvoidanceEnv()
    import time

    start_time = time.time()
    env.reset()
    # env.render()
    for _ in range(25000):
        print(_)
        a = env.action_space_cart.sample()
        # a = np.array([0, 0])
        obs, reward, done, infos = env.step(a)
        env.render()
        if done:
            env.reset()
            # env.render()
    print("Test loop took: ", (time.time() - start_time) / 250)
