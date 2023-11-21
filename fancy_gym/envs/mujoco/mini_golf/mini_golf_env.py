import os
from typing import Optional

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

import mujoco

MAX_EPISODE_STEPS_MINI_GOLF = 100

# CTXT SPACE BOUNDS:[[min_X_RED, min_Y_RED, min_X_GREEN, min_Y_GREEN, min_goal_width],
#                    [max_X_RED, max_Y_RED, max_X_GREEN, max_Y_GREEN, max_goal_width]]

# MIN_GOAL_WIDTH = 0.06
# MAX_GOAL_WIDTH = 0.3
# CONTEXT_BOUNDS = np.array([[0.19, -0.025, 0.3, -0.5, MIN_GOAL_WIDTH], [0.65, 0.2, 0.6, -0.1, MAX_GOAL_WIDTH]])


# Red obstacle X- and Y Variation
MIN_X_RED = 0.19
MAX_X_RED = 0.65
MIN_Y_RED = -0.025
MAX_Y_RED = 0.2

# Green obstacle X- and Y Variation
MIN_X_GREEN = 0.3
MAX_X_GREEN = 0.6
MIN_Y_GREEN = -0.5
MAX_Y_GREEN = -0.1

# Initial Ball X-Variation
MIN_X_BALL_POS = 0.25
MAX_X_BALL_POS = 0.6

# # Gaol Wall Shift Variation
# MAX_WALL_SHIFT = 0.245

MIN_X_GOAL_POS = 0.18
MAX_X_GOAL_POS = 0.67

CONTEXT_BOUNDS = np.array([[MIN_X_RED, MIN_Y_RED, MIN_X_GREEN, MIN_Y_GREEN, MIN_X_BALL_POS, MIN_X_GOAL_POS],
                           [MAX_X_RED, MAX_Y_RED, MAX_X_GREEN, MAX_Y_GREEN, MAX_X_BALL_POS, MAX_X_GOAL_POS]])

CONTEXT_BOUNDS_ONE_OBS = np.array([[MIN_X_GREEN, MIN_Y_GREEN, MIN_X_BALL_POS, MIN_X_GOAL_POS],
                                   [MAX_X_GREEN, MAX_Y_GREEN, MAX_X_BALL_POS, MAX_X_GOAL_POS]])


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
        self._q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self._q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self._init_qpos = np.array([0., 0., 0., -1.5, 0., 1.5, 0., 0., 0., 0.425, 0.4, 0.005, 1, 0, 0, 0])
        self._des_robot_qpos = np.array([0.90138834, 0.94911663, 0.15175606, -1.0552696, -0.13568928, 1.99531533,
                                         1.82624438])
        self._init_qvel = np.zeros(15)
        self._init_xpos_goal_wall_left = 0.0
        self._init_xpos_goal_wall_right = 0.85
        self.frame_skip = frame_skip
        self._episode_energy = 0.
        self._had_ball_contact = False
        self._passed_goal = False
        self._pass_threshold = -0.75
        self._id_set = False
        self._ball_traj = []
        self._rod_traj = []
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", xml_name),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

    def _set_ids(self):
        self._ball_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_contact")
        self._rod_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rod_contact")
        self._rod2_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rod_contact2")
        self._id_set = True

    def step(self, action):
        if not self._id_set:
            self._set_ids()

        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(action + self.data.qfrc_bias[:7].copy(), -self.q_torque_max, self.q_torque_max)

        unstable_simulation = False

        for _ in range(self.frame_skip):
            try:
                self.do_simulation(resultant_action, 1)
            except Exception as e:
                print(e)
                unstable_simulation = True
                break

            if not self._had_ball_contact:
                self._had_ball_contact = (self._contact_checker(self._ball_contact_id, self._rod_contact_id)
                                          or self._contact_checker(self._ball_contact_id, self._rod2_contact_id))

            else:
                if not self._passed_goal:
                    # check if ball has passed goal (ball pos - radius):
                    # y-wall position is at -0.7-> Smaller Y-Values mean ball has passed the goal
                    # we are adding a threshold to make sure that the ball has passed the -0.7. Mujoco has soft
                    # constraints which might lead to temporary "wall breakthroughs"
                    self._passed_goal = self.data.body("ball").xpos.copy()[1] - 0.025 < self._pass_threshold

            self._ball_traj.append(self.data.body("ball").xpos.copy())
            self._rod_traj.append(self.data.site("rod_tip").xpos.copy())
        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if (self._steps >= MAX_EPISODE_STEPS_MINI_GOLF or unstable_simulation) else False

        ball_pos = self.data.body("ball").xpos.copy()
        # red_obs_box = self.data.body("obstacle_box_0").xpos.copy()
        # green_obs_box = self.data.body("obstacle_box_1").xpos.copy()

        rod_tip_pos = self.data.site("rod_tip").xpos.copy()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, action)
        else:
            reward = -50

        obs = self._get_obs()
        rod_tip_ball_dist = np.linalg.norm(rod_tip_pos - ball_pos)
        infos = {
            'episode_end': episode_end,
            'ball_pos': ball_pos,
            'ball_goal_y_pos': ball_pos[1],
            'rod_tip_ball_dist': rod_tip_ball_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'is_success': self._passed_goal,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

    def _get_reward(self, episode_end, pol_action):
        if not episode_end:
            return -0.0005 * np.sum(np.square(pol_action))
        min_b_rod_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._rod_traj), axis=1))
        if not self._had_ball_contact:
            return 0.2 * (1 - np.tanh(min_b_rod_dist))
        if not self._passed_goal:
            min_y_val = np.min(np.array(self._ball_traj)[:, 1])
            goal_pos = self.data.site("ball_target").xpos.copy()
            min_ball_goal_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - goal_pos[None, :], axis=1))
            return 2 * (1 - np.tanh(min_ball_goal_dist)) + 0.5 * (1 - np.tanh(min_y_val - self._pass_threshold))
        return 6

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
        # randomly sample obstacles
        positions = self.sample_context()
        red_obs_pos = positions[:3]
        green_obs_pos = positions[3:6]
        init_ball_pos = positions[6:9]
        goal_pos = positions[-1]

        self._init_qpos[9:12] = init_ball_pos
        self.set_state(self._init_qpos, self._init_qvel)

        self.model.body('obstacle_box_0').pos = red_obs_pos
        self.model.body('obstacle_box_1').pos = green_obs_pos

        # This was for varying wall widths, but this turned out to be trivial:
        # # get delta_x to the left and right wall x-positions:
        # delta_x = (goal_width - MIN_GOAL_WIDTH) / 2
        # self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left - delta_x
        # self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + delta_x

        # self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left + wall_shift
        # self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + wall_shift

        # self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.425
        self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.428
        # self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.425
        self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.428

        self.model.site("ball_target").pos[0] = goal_pos

        # desired_tcp_pos = self.data.body("ball").xpos.copy() + np.array([0.0, 0.25, 0.15])
        # desired_tcp_quat = np.array([0, 1, 0, 0])
        # desired_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)
        # print(desired_joint_pos)
        # self.data.qpos[:7] = desired_joint_pos

        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._passed_goal = False
        self._had_ball_contact = False
        self._ball_traj = []
        self._rod_traj = []
        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=CONTEXT_BOUNDS[0], high=CONTEXT_BOUNDS[1])
        red_box_pos = np.append(pos[:2], [0.])
        green_box_pos = np.append(pos[2:4], [0.])
        init_ball_pos = np.append(pos[4], [0.5, 0.005])
        goal_pos = pos[-1]
        return np.concatenate([red_box_pos, green_box_pos, init_ball_pos, [goal_pos]])

    def set_context(self, context):
        # rest box to initial position
        red_obs_pos = np.append(context[:2], [0])
        green_obs_pos = np.append(context[2:4], [0])
        init_ball_pos = np.append([context[4]], [0.5, 0.005])
        goal_pos = context[-1]

        self._init_qpos[9:12] = init_ball_pos
        self.set_state(self._init_qpos, self._init_qvel)

        self.model.body('obstacle_box_0').pos = red_obs_pos
        self.model.body('obstacle_box_1').pos = green_obs_pos

        # This was for varying wall widths, but this turned out to be trivial:
        # # get delta_x to the left and right wall x-positions:
        # delta_x = (goal_width - MIN_GOAL_WIDTH) / 2
        # self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left - delta_x
        # self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + delta_x

        # self.model.geom("goal_wall_left").pos[0] = self._init_xpos_goal_wall_left + wall_shift
        # self.model.geom("goal_wall_right").pos[0] = self._init_xpos_goal_wall_right + wall_shift

        # self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.425
        self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.428
        # self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.425
        self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.428

        self.model.site("ball_target").pos[0] = goal_pos

        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._passed_goal = False
        self._had_ball_contact = False
        self._ball_traj = []
        self._rod_traj = []
        return self._get_obs()

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]
            if (con.geom1 == id_1 and con.geom2 == id_2) or (con.geom1 == id_2 and con.geom2 == id_1):
                return True
        return False

    def _calc_current_goal_width(self):
        return self.model.geom("goal_wall_right").pos[0].copy() - self.model.geom("goal_wall_left").pos[
            0].copy() - 2 * 0.125

    # TODO: For step based envs the observation space needs to be adjusted
    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            self.data.body("ball").xpos.copy(),  # position of ball
            self.data.body("obstacle_box_0").xpos.copy(),  # position of red obstacle
            self.data.body("obstacle_box_1").xpos.copy(),  # position of green obstacle
            [self._calc_current_goal_width()]  # Current width of the target wall
        ])
        return obs

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
            q = np.clip(q, self._q_min, self._q_max)
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
            qd_null_limit_max = pgain_limit * (self._q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (self._q_min + margin_to_limit - q)
            qd_null_limit[q > self._q_max - margin_to_limit] += qd_null_limit_max[q > self._q_max - margin_to_limit]
            qd_null_limit[q < self._q_min + margin_to_limit] += qd_null_limit_min[q < self._q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q

    def check_traj_validity(self, action, pos_traj, vel_traj, tau_bound, delay_bound):
        time_invalid = action[0] > tau_bound[1] or action[0] < tau_bound[0]
        if time_invalid or np.any(pos_traj > self._q_max) or np.any(pos_traj < self._q_min):
            return False, pos_traj, vel_traj
        return True, pos_traj, vel_traj

    def _get_traj_invalid_penalty(self, action, pos_traj, tau_bound, delay_bound):
        tau_invalid_penalty = 3 * (np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]]))
        violate_high_bound_error = np.mean(np.maximum(pos_traj - self._q_max, 0))
        violate_low_bound_error = np.mean(np.maximum(self._q_min - pos_traj, 0))
        invalid_penalty = tau_invalid_penalty + violate_high_bound_error + violate_low_bound_error
        return -20 * invalid_penalty - 5

    def get_invalid_traj_step_return(self, action, pos_traj, contextual_obs, tau_bound, delay_bound):
        obs = self._get_obs() if contextual_obs else np.concatenate(
            [self._get_obs(), np.array([0])])  # 0 for invalid traj
        penalty = self._get_traj_invalid_penalty(action, pos_traj, tau_bound, delay_bound)
        return obs, penalty, True, {
            'episode_end': [True],
            'ball_pos': [self.data.body("ball").xpos.copy()],
            'ball_goal_y_pos': [self.data.body("ball").xpos.copy()[1]],
            'rod_tip_ball_dist': [10],
            'episode_energy': [self._episode_energy],
            'is_success': [self._passed_goal],
            'num_steps': [self._steps]
        }


class MiniGolfQuadRewEnv(MiniGolfEnv):
    def _get_reward(self, episode_end, pol_action):
        if not episode_end:
            return -0.0005 * np.sum(np.square(pol_action))
        min_b_rod_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._rod_traj), axis=1))
        if not self._had_ball_contact:
            return -2 * min_b_rod_dist - 2
        if not self._passed_goal:
            ball_contact_bonus = 5
            min_y_val = np.min(np.array(self._ball_traj)[:, 1])
            goal_pos = self.data.site("ball_target").xpos.copy()
            min_ball_goal_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - goal_pos[None, :], axis=1))
            return ball_contact_bonus - 5 * min_ball_goal_dist - 2 * min_y_val
        return 8


class MiniGolfOneObsEnv(MiniGolfEnv):

    def reset_model(self):
        # randomly sample obstacles
        positions = self.sample_context()
        green_obs_pos = positions[:3]
        init_ball_pos = positions[3:6]
        goal_pos = positions[-1]

        self._init_qpos[9:12] = init_ball_pos
        self.set_state(self._init_qpos, self._init_qvel)

        self.model.body('obstacle_box_1').pos = green_obs_pos
        self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.428
        self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.428

        self.model.body('obstacle_box_0').pos = np.array([5, 5, 5])

        self.model.site("ball_target").pos[0] = goal_pos
        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._passed_goal = False
        self._had_ball_contact = False
        self._ball_traj = []
        self._rod_traj = []
        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=CONTEXT_BOUNDS_ONE_OBS[0], high=CONTEXT_BOUNDS_ONE_OBS[1])
        green_box_pos = np.append(pos[:2], [0.])
        init_ball_pos = np.append(pos[2], [0.5, 0.005])
        goal_pos = pos[-1]
        return np.concatenate([green_box_pos, init_ball_pos, [goal_pos]])

    def set_context(self, context):
        # rest box to initial position
        green_obs_pos = np.append(context[:2], [0])
        init_ball_pos = np.append([context[2]], [0.5, 0.005])
        goal_pos = context[-1]

        self._init_qpos[9:12] = init_ball_pos
        self.set_state(self._init_qpos, self._init_qvel)

        self.model.body('obstacle_box_1').pos = green_obs_pos
        self.model.geom("goal_wall_left").pos[0] = goal_pos - 0.428
        self.model.geom("goal_wall_right").pos[0] = goal_pos + 0.428

        self.model.body('obstacle_box_0').pos = np.array([5, 5, 5])

        self.model.site("ball_target").pos[0] = goal_pos

        self.data.qpos[:7] = self._des_robot_qpos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.
        self._passed_goal = False
        self._had_ball_contact = False
        self._ball_traj = []
        self._rod_traj = []
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            self.data.body("ball").xpos.copy(),  # position of ball
            self.data.body("obstacle_box_1").xpos.copy(),  # position of green obstacle
            [self._calc_current_goal_width()]  # Current width of the target wall
        ])
        return obs

if __name__ == '__main__':
    # env = MiniGolfEnv()
    # env = MiniGolfQuadRewEnv()
    env = MiniGolfOneObsEnv()
    import time

    start_time = time.time()
    obs = env.reset()
    env.render()
    for _ in range(5000):
        obs, reward, done, infos = env.step(env.action_space.sample())
        # env.reset()
        env.render()
        if done:
            env.reset()
            print(reward)
