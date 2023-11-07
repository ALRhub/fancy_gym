import os
from typing import Optional

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

import mujoco

MAX_EPISODE_STEPS_MINI_GOLF = 100

# CTXT SPACE BOUNDS:[[min_X_RED, min_Y_RED, min_X_GREEN, min_Y_GREEN], [max_X_RED, max_Y_RED, max_Y_GREEN, max_Y_GREEN]]
BOX_POS_BOUND = np.array([[0.3, -0.025, 0.3, -0.5], [0.6, 0.2, 0.6, -0.1]])


class MiniGolfEnv(MujocoEnv, utils.EzPickle):
    """
    franka box mini golf  environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    """

    def __init__(self, frame_skip: int = 10, xml_name="box_pushing.xml", **kwargs):
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.init_qpos_box_pushing = np.array([0., 0., 0., -1.5, 0., 1.5, 0., 0., 0., 0.6, 0.45, 0.0, 1., 0., 0., 0.])
        self.desired_qpos_robot = np.array([[0.38706806, 0.17620842, 0.24989142, -2.39914377, -0.07986905, 2.56857367,
                                             1.47951693]])
        self.init_qvel_box_pushing = np.zeros(15)
        self.box_init_pos = np.array([0.4, 0.3, -0.01, 0.0, 0.0, 0.0, 1.0])
        self.frame_skip = frame_skip

        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max
        self._desired_rod_quat = desired_rod_quat

        self._episode_energy = 0.
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", xml_name),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

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
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        box_pos = self.data.body("box_0").xpos.copy()
        box_quat = self.data.body("box_0").xquat.copy()
        target_pos = self.data.body("replan_target_pos").xpos.copy()
        target_quat = self.data.body("replan_target_pos").xquat.copy()
        target_quat2 = self.data.body("replan_target_pos2").xquat.copy()
        target_quat3 = self.data.body("replan_target_pos3").xquat.copy()
        target_quat4 = self.data.body("replan_target_pos4").xquat.copy()
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, box_pos, box_quat, target_pos, target_quat, target_quat2,
                                      target_quat3, target_quat4, rod_tip_pos, rod_quat, qpos, qvel, action)
        else:
            reward = -50

        obs = self._get_obs()
        box_goal_pos_dist = 0. if not episode_end else np.linalg.norm(box_pos - target_pos)
        box_goal_quat_dist = 0. if not episode_end else self._get_rotation_dist(box_quat, target_quat, target_quat2,
                                                                                target_quat3, target_quat4)
        infos = {
            'episode_end': episode_end,
            'box_pos': box_pos,
            'box_or': quat2euler(box_quat),
            'box_goal_pos_dist': box_goal_pos_dist,
            'box_goal_rot_dist': box_goal_quat_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'is_success': True if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5 else False,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

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

    def reset_model(self):
        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)

        self.data.joint("box_joint").qpos = self.box_init_pos

        # set target position
        box_target_pos = self.sample_context()
        while np.linalg.norm(box_target_pos[:2] - self.box_init_pos[:2]) < 0.3:
            box_target_pos = self.sample_context()
        # box_target_pos[0] = 0.4
        # box_target_pos[1] = -0.3
        # box_target_pos[-4:] = np.array([0.0, 0.0, 0.0, 1.0])
        self.model.body_pos[2] = box_target_pos[:3]
        self.model.body_quat[2] = box_target_pos[-4:]
        self.model.body_pos[3] = box_target_pos[:3]
        self.model.body_quat[3] = box_target_pos[-4:]

        ### Avoid calculating the IK all the time:
        # # set the robot to the right configuration (rod tip in the box)
        # desired_tcp_pos = box_init_pos[:3] + np.array([0.0, 0.0, 0.15])
        # desired_tcp_quat = np.array([0, 1, 0, 0])
        # # desired_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)
        # # self.data.qpos[:7] = desired_joint_pos
        self.data.qpos[:7] = self.desired_qpos_robot

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.

        return self._get_obs()

    def sample_context(self):
        # pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        lb = np.append(BOX_POS_BOUND[0], [-0.01])
        ub = np.append(BOX_POS_BOUND[1], [-0.01])
        pos = self.np_random.uniform(low=lb, high=ub)
        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        return np.concatenate([pos, quat])

    def _get_pi_half_variants(self, angle_in_rad):
        angle2 = angle_in_rad + np.pi / 2
        angle3 = angle_in_rad + np.pi
        angle4 = angle_in_rad + 1.5 * np.pi
        return angle2, angle3, angle4

    def _get_pi_half_variant_quats(self, angle, angle2, angle3, angle4):
        quat_or_context = rot_to_quat(angle, np.array([0, 0, 1]))
        quat_or_context2 = rot_to_quat(angle2, np.array([0, 0, 1]))
        quat_or_context3 = rot_to_quat(angle3, np.array([0, 0, 1]))
        quat_or_context4 = rot_to_quat(angle4, np.array([0, 0, 1]))
        return quat_or_context, quat_or_context2, quat_or_context3, quat_or_context4

    def _set_target_box_pos_and_quat(self, target_xy_box_pos, quat_target_box, quat_target_box2, quat_target_box3,
                                     quat_target_box4):
        self.model.body_pos[2][:2] = target_xy_box_pos
        self.model.body_pos[2][-1] = -0.01
        self.model.body_quat[2] = quat_target_box

        self.model.body_pos[3][:2] = target_xy_box_pos
        self.model.body_pos[3][-1] = -0.01
        self.model.body_quat[3] = quat_target_box

        self.model.body_pos[4][:2] = target_xy_box_pos
        self.model.body_pos[4][-1] = -0.01
        self.model.body_quat[4] = quat_target_box2

        self.model.body_pos[5][:2] = target_xy_box_pos
        self.model.body_pos[5][-1] = -0.01
        self.model.body_quat[5] = quat_target_box3

        self.model.body_pos[6][:2] = target_xy_box_pos
        self.model.body_pos[6][-1] = -0.01
        self.model.body_quat[6] = quat_target_box4

    def set_context(self, context):
        angle = context[-1]
        angle2, angle3, angle4 = self._get_pi_half_variants(angle)
        quat_or_context, quat_or_context2, quat_or_context3, quat_or_context4 = self._get_pi_half_variant_quats(angle,
                                                                                                                angle2,
                                                                                                                angle3,
                                                                                                                angle4)

        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)
        self.data.joint("box_joint").qpos = self.box_init_pos

        self._set_target_box_pos_and_quat(context[:2], quat_or_context, quat_or_context2, quat_or_context3,
                                          quat_or_context4)

        self.data.qpos[:7] = self.desired_qpos_robot

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        return self._get_obs()

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat, target_quat2, target_quat3,
                    target_quat4, rod_tip_pos, rod_quat, qpos, qvel, action):
        raise NotImplementedError

    def _get_min_rot_dist(self, box_quat, target_quat, target_quat2, target_quat3, targetquat4):
        rot_dist = rotation_distance(box_quat, target_quat)
        rot_dist2 = rotation_distance(box_quat, target_quat2)
        rot_dist3 = rotation_distance(box_quat, target_quat3)
        rot_dist4 = rotation_distance(box_quat, targetquat4)
        return np.min(np.array([rot_dist, rot_dist2, rot_dist3, rot_dist4]))

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            # self.data.qfrc_bias[:7].copy(),  # joint gravity compensation
            # self.data.site("rod_tip").xpos.copy(),  # position of rod tip
            # self.data.body("push_rod").xquat.copy(),  # orientation of rod
            self.data.body("box_0").xpos.copy(),  # position of box
            self.data.body("box_0").xquat.copy(),  # orientation of box
            self.data.body("replan_target_pos").xpos.copy(),  # position of target
            self.data.body("replan_target_pos").xquat.copy()  # orientation of target
        ])
        return obs

    def _joint_limit_violate_penalty(self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False):
        penalty = 0.
        p_coeff = 1.
        v_coeff = 1.
        # q_limit
        if enable_pos_limit:
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (abs(np.sum(higher_error[qpos > self._q_max])) +
                                  abs(np.sum(lower_error[qpos < self._q_min])))
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_error > 0.]))
        return penalty

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

    def _get_rotation_dist(self, box_quat, target_quat, target_quat2, target_quat3, target_quat4):
        raise NotImplementedError


class BoxPushingDense(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, **kwargs):
        super(BoxPushingDense, self).__init__(frame_skip=frame_skip, xml_name="box_pushing.xml")

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat, target_quat2, target_quat3,
                    target_quat4, rod_tip_pos, rod_quat, qpos, qvel, action):
        joint_penalty = self._joint_limit_violate_penalty(qpos,
                                                          qvel,
                                                          enable_pos_limit=True,
                                                          enable_vel_limit=True)
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        energy_cost = -0.0005 * np.sum(np.square(action))

        reward = joint_penalty + tcp_box_dist_reward + \
                 box_goal_pos_dist_reward + box_goal_rot_dist_reward + energy_cost

        rod_inclined_angle = rotation_distance(rod_quat, self._desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        return reward

    def _get_rotation_dist(self, box_quat, target_quat, target_quat2, target_quat3, target_quat4):
        return rotation_distance(box_quat, target_quat)



if __name__ == '__main__':
    # env = BoxPushingDense()
    env = BoxPushingTemporalSparseNotInclinedInit()
    import time

    start_time = time.time()
    box_target_positions = []
    obs = env.reset()
    env.render()
    for _ in range(5000):
        obs = env.reset()
        env.render()
    #     a = env.action_space.sample()
    #     obs, reward, episode_end, infos = env.step(a)
    #     if episode_end:
    #         obs = env.reset()
    #         episode_end = False
    #     box_target_positions.append(np.array([obs[-7], obs[-6]]))
    # # # env.render()
    # for _ in range(10000):
    #     a = env.action_space.sample()
    #     obs, reward, done, infos = env.step(a)
    #     if done:
    #         obs=env.reset()
    #         box_target_positions.append(np.array([obs[-7], obs[-6]]))
    #         # env.render()
    # print('Test loop took: ', (time.time() - start_time) / 100)

    # import matplotlib.pyplot as plt
    #
    # box_target_positions = np.array(box_target_positions)
    # plt.figure()
    # plt.scatter(box_target_positions[:, 0], box_target_positions[:, 1])
