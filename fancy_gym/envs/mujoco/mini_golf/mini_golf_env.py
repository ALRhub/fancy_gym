import os
from typing import Optional

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

import mujoco

MAX_EPISODE_STEPS_MINI_GOLF = 100

# CTXT SPACE BOUNDS:[[min_X_RED, min_Y_RED, min_X_GREEN, min_Y_GREEN, min_goal_width],
#                    [max_X_RED, max_Y_RED, max_Y_GREEN, max_Y_GREEN, max_goal_width]]

MIN_GOAL_WIDTH = 0.06
MAX_GOAL_WIDTH = 0.3
BOX_POS_BOUND = np.array([[0.19, -0.025, 0.3, -0.5, MIN_GOAL_WIDTH], [0.65, 0.2, 0.6, -0.1, MAX_GOAL_WIDTH]])


def skew(x):
    """
    Returns the skew-symmetric matrix of x
    param x: 3x1 vector
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    param curr_quat: current quaternion
    param des_quat: desired quaternion
    return: difference between current quaternion and desired quaternion
    """
    return curr_quat[0] * des_quat[1:] - des_quat[0] * curr_quat[1:] - skew(des_quat[1:]) @ curr_quat[1:]


class MiniGolfEnv(MujocoEnv, utils.EzPickle):
    """
    franka box mini golf  environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    """

    def __init__(self, frame_skip: int = 10, xml_name="mini_golf.xml", **kwargs):
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.q_torque_max = np.array([90., 90., 90., 90., 12., 12., 12.])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self._init_qpos = np.array([0., 0., 0., -1.5, 0., 1.5, 0., 0., 0., 0.425, 0.5, 0.005, 1, 0, 0, 0])
        self._des_robot_qpos = np.array([0.82571042, 0.81006124, 0.17260485, -1.37485921, -0.15166258, 2.17198979,
                                         1.81153316])
        self._init_qvel = np.zeros(15)
        self._is_success = False
        self._init_xpos_goal_wall_left = 0.27
        self._init_xpos_goal_wall_right = 0.58
        self.frame_skip = frame_skip
        self._episode_energy = 0.
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", xml_name),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

    def step(self, action):
        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(action + self.data.qfrc_bias[:7].copy(), -self.q_torque_max, self.q_torque_max)

        unstable_simulation = False

        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_MINI_GOLF else False

        ball_pos = self.data.body("ball").xpos.copy()
        red_obs_box = self.data.body("obstacle_box_0").xpos.copy()
        green_obs_box = self.data.body("obstacle_box_1").xpos.copy()
        ball_target_pos = self.data.site("ball_target").xpos.copy()

        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()

        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, ball_pos, red_obs_box, green_obs_box, ball_target_pos,
                                      rod_tip_pos, rod_quat, qpos, qvel, action)
        else:
            reward = -50

        obs = self._get_obs()
        ball_goal_dist = np.linalg.norm(ball_target_pos - ball_pos)
        self._is_success = True if (self._is_success or ball_pos[1] < ball_target_pos[1]) else False
        rod_tip_ball_dist = np.linalg.norm(rod_tip_pos - ball_pos)
        infos = {
            'episode_end': episode_end,
            'ball_pos': ball_pos,
            'ball_goal_dist': ball_goal_dist,
            'rod_tip_ball_dist': rod_tip_ball_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'is_success': self._is_success,
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

    """
    goal width context: varying the goal width makes the task easier/harder, but helps the agent to learn from easier
                        tasks first. Setting the goal width follows the following equation:
                        delta_x = (goal_width - MIN_GOAL_WIDTH)/2 
                        -> x-position of right wall: 0.58 + delta_x
                        -> x-position of left wall: 0.27 - delta_x 
                        I.e. a desired width of MIN_GOAL_WIDTH results in delta_x = 0.0.  
    """

    def reset_model(self):
        # rest box to initial position
        self.set_state(self._init_qpos, self._init_qvel)

        # randomly sample obstacles
        positions = self.sample_context()
        red_obs_pos = positions[:3]
        green_obs_pos = positions[-4:-1]
        goal_width = positions[-1]

        self.model.body('obstacle_box_0').pos = red_obs_pos
        self.model.body('obstacle_box_1').pos = green_obs_pos

        # get delta_x to the left and right wall x-positions:
        delta_x = (goal_width - MIN_GOAL_WIDTH) / 2
        self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left - delta_x
        self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + delta_x

        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._is_success = False

        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        red_box_pos = np.append(pos[:2], [0.])
        green_box_pos = np.append(pos[-3:-1], [0.])
        goal_width = pos[-1]
        return np.concatenate([red_box_pos, green_box_pos, [goal_width]])

    def set_context(self, context):
        # rest box to initial position
        self.set_state(self._init_qpos, self._init_qvel)

        red_obs_pos = np.append(context[:2], [0])
        green_obs_pos = np.append(context[-4:-1], [0])
        goal_width = context[-1]

        self.model.body('obstacle_box_0').pos = red_obs_pos
        self.model.body('obstacle_box_1').pos = green_obs_pos

        # get delta_x to the left and right wall x-positions:
        delta_x = (goal_width - MIN_GOAL_WIDTH) / 2
        self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left - delta_x
        self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + delta_x

        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._is_success = False
        return self._get_obs()

    def _get_reward(self, episode_end, ball_pos, red_obs_box, green_obs_box, ball_target_pos, rod_tip_pos,
                    rod_quat, qpos, qvel, action):
        return 0

    def _calc_current_goal_width(self):
        return self.model.geom("goal_wall_right").pos[0].copy() - self.model.geom("goal_wall_left").pos[
            0].copy() - 2 * 0.125

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            self.data.body("ball").xpos.copy(),  # position of box
            self.data.body("obstacle_box_0").xpos.copy(),  # position of red obstacle
            self.data.body("obstacle_box_1").xpos.copy(),  # position of green obstacle
            [self._calc_current_goal_width()]              # Current width of the target wall
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
            q = np.clip(q, self.q_min, self.q_max)
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
            qd_null_limit_max = pgain_limit * (self.q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (self.q_min + margin_to_limit - q)
            qd_null_limit[q > self.q_max - margin_to_limit] += qd_null_limit_max[q > self.q_max - margin_to_limit]
            qd_null_limit[q < self.q_min + margin_to_limit] += qd_null_limit_min[q < self.q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q


if __name__ == '__main__':
    env = MiniGolfEnv()
    import time

    start_time = time.time()
    obs = env.reset()
    env.render()
    for _ in range(5000):
        # obs, reward, done, infos = env.step(env.action_space.sample())
        env.reset()
        env.render()
        # if done:
        #     env.reset()
