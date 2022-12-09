import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
import mujoco
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
    rot_to_quat,
    get_quaternion_error,
    q_max, q_min,
    q_dot_max, q_torque_max,
)
from fancy_gym.black_box.controller.pd_controller import PDController

MAX_EPISODE_STEPS_BOX_PUSHING_BIN = 1000
BOX_INIT_FRAME_SKIPS = 500  # boxes need time to fall
PUSH_DISTANCE = 0.03  # 3cm ~ 1 / 3 of box sizes

# Need to set by hand depending on the environment config in the xml files
MAX_NUM_BOXES = 10
BOX_POS_BOUND = np.array([[0.4, -0.3, 0.06], [0.8, 0.3, 0.45]])
BIN_SIZE = 0.2
NUM_BINS = 3
# (ahead, lean 45 back, ahead, lean 225 front, ahead, look down, camera front)
START_POS = np.array([0, -np.pi/8, 0.0, -np.pi*5/8, 0.0, np.pi/2, np.pi/4])
ROBOT_CENTER = np.array([0.16, 0.0])
ROBOT_RADIUS = 0.788


class BoxPushingBin(MujocoEnv, utils.EzPickle):
    """
    franka box pushing environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    observation space:
    """
    def __init__(
        self,
        num_boxes: int = 10,
        frame_skip: int = 10,
        width: int = 244,
        height: int = 244,
    ):
        assert num_boxes <= MAX_NUM_BOXES
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.frame_skip = frame_skip
        self.num_boxes = num_boxes
        self._q_max, self._q_min, self._q_dot_max = q_max, q_min, q_dot_max
        self.width, self.height = width, height

        self.init_qpos_box_pushing = np.zeros(7 + MAX_NUM_BOXES * 7)
        self.init_qvel_box_pushing = np.zeros(7 + MAX_NUM_BOXES * 6)
        self.boxes = ["box_" + str(i) for i in range(num_boxes)]
        self.joints = ["box_joint_" + str(i) for i in range(num_boxes)]
        self.hidden = ["box_joint_" + str(i) for i in range(num_boxes, MAX_NUM_BOXES)]
        self.boxes_out_bins = np.arange(num_boxes)

        # noise of 8deg ~ pi/21rad, ensure >95% values inside the range with 3sigma=range
        self.noisy_start_pos = lambda : np.clip(
            START_POS + np.random.normal(0, np.pi / 21 / 3, START_POS.shape),
            self._q_min,
            self._q_max
        )
        self.robot_tcp_penalty = lambda x :\
            (np.linalg.norm(x - ROBOT_CENTER) > ROBOT_RADIUS) * -100
        self.controller = PDController()

        self._episode_energy = 0.
        MujocoEnv.__init__(
            self,
            model_path=os.path.join(
                os.path.dirname(__file__),
                "assets/box_pushing_bins.xml",
            ),
            frame_skip=self.frame_skip,
            mujoco_bindings="mujoco"
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

        # Boxes that are extra are rendered inside the bins
        for i, joint in enumerate(self.hidden):
            x, y, z = i % 18 // 6, i % 6, i // 18  # arange in a grid
            start_idx = 7 * (1 + i + self.num_boxes)
            self.init_qpos_box_pushing[start_idx:start_idx + 7] = np.array(
                [1.21 + x * 0.12, -0.58 + 0.23 * y , 0.2 + z * 0.12, 0, 0, 0, 0]
            )

        # camera calibration utilities
        fovys = [self.model.cam("rgbd").fovy[0], self.model.cam("rgbd_cage").fovy[0]]
        focal = 0.5 * self.height / np.tan(np.array(fovys) * np.pi / 360)
        self.focal_mat = {
            "rgbd": np.array([
                [-focal[0], 0, self.width / 2.0, 0],
                [0, focal[0], self.height / 2.0, 0],
                [0, 0, 1, 0]
            ]),
            "rgbd_cage": np.array([
                [-focal[1], 0, self.width / 2.0, 0],
                [0, focal[1], self.height / 2.0, 0],
                [0, 0, 1, 0]
            ]),
        }

        self.near = self.model.vis.map.znear * self.model.stat.extent
        self.far = self.model.vis.map.zfar * self.model.stat.extent

    def step(self, action):
        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(
            action + self.data.qfrc_bias[:7].copy(),
            -q_torque_max,
            q_torque_max
        )
        unstable_simulation = False

        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))
        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING_BIN else False

        box_pos = [self.data.body(box).xpos.copy() for box in self.boxes]
        box_pos_xyz = [b[:3] for b in box_pos]
        box_quat = [self.data.body(box).xquat.copy() for box in self.boxes]
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        if not unstable_simulation:
            reward = self._get_reward(action, qpos, qvel, box_pos_xyz)
        else:
            reward = -50

        obs = self._get_obs()
        infos = {
            'episode_end': episode_end,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

    def reset_model(self):
        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)

        # Initialize box positions randomly, ensure collision free init by trial
        positions = []
        for joint in self.joints:
            new_pos, collision = self.sample_context(), True
            while collision and len(positions) > 0:
                for p in positions:
                    collision = np.linalg.norm(new_pos[:3] - p[:3]) < 0.15
                    if collision:  # collision detected sample new pos and check again
                        new_pos = self.sample_context()
                        break
            self.data.joint(joint).qpos = new_pos
            positions.append(new_pos)
        self.boxes_out_bins = np.arange(len(self.boxes))  # assume all boxes out

        # Robot out of the way of boxes before dropping them
        self.data.qpos[:7] = START_POS + np.array([0, 0, 0, np.pi*5/8, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)
        self._steps, self._episode_energy = 0, 0

        # Init environemnt by letting boxes fall
        no_action = np.clip(
            np.zeros(self.action_space.shape) + self.data.qfrc_bias[:7].copy(),
            -q_torque_max,
            q_torque_max
        )
        self.do_simulation(no_action, BOX_INIT_FRAME_SKIPS)

        self.boxes_out_bins = np.delete(  # Remove boxes that fell in bin after box init
            self.boxes_out_bins,
            self.boxes_in_bin(
                np.array([self.data.body(box).xpos.copy() for box in self.boxes])\
                    [self.boxes_out_bins]
            )
        )
        self.reset_robot_pos()

        return self._get_obs()

    def reset_robot_pos(self):
        self.data.qpos[:7] = self.noisy_start_pos()
        self.data.qvel[:7] = np.zeros(START_POS.shape)
        mujoco.mj_forward(self.model, self.data)

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        quat = rot_to_quat(theta, np.array([1, 1, 1]))
        return np.concatenate([pos, quat])

    def _get_reward(self, action, qpos, qvel, box_pos=None):
        """
        By default the environment should learn smooth movement with the least torque
        necessary without violating constraints.

        Args:
            action (np.array): action taken during step
            qpos (np.array): robot position, angle for each joint
            qvel (np.array): robot velocity, torque for each joint
        Return:
            (float): scalar reward value
        """
        joint_penalty_reward = self._joint_limit_violate_penalty(
            qpos,
            qvel,
            enable_pos_limit=True,
            enable_vel_limit=True
        )
        energy_cost = -0.0005 * np.sum(np.square(action))

        return joint_penalty_reward + energy_cost

    def _get_obs(self):
        box_pos = [self.data.body(box).xpos.copy() for box in self.boxes]
        box_quat = [self.data.body(box).xquat.copy() for box in self.boxes]
        obs = np.concatenate([
                self.data.qpos[:7].copy(),  # joint position
                self.data.qvel[:7].copy(),  # joint velocity
                self.data.qfrc_bias[:7].copy(),  # joint gravity compensation
                self.data.site("rod_tip").xpos.copy(),  # position of rod tip
                self.data.body("push_rod").xquat.copy(),  # orientation of rod
            ] + box_pos + box_quat
        )
        return obs

    def _joint_limit_violate_penalty(
        self,
        qpos,
        qvel,
        enable_pos_limit=False,
        enable_vel_limit=False
    ):
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
        eps = 1e-5          # threshold for convergence
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

            if (np.linalg.norm(current_cart_quat - desired_cart_quat) >
                np.linalg.norm(current_cart_quat + desired_cart_quat)):
                current_cart_quat = -current_cart_quat
            cart_quat_error = np.clip(
                get_quaternion_error(current_cart_quat, desired_cart_quat),
                -0.5,
                0.5
            )

            err = np.hstack((cart_pos_error, cart_quat_error))
            err_norm = np.sum(cart_pos_error**2) +\
                np.sum((current_cart_quat - desired_cart_quat)**2)
            if err_norm > old_err_norm:
                q = q_old
                dt = 0.7 * dt
                continue
            else:
                dt = 1.025 * dt

            if err_norm < eps or i > IT_MAX:
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
            qd_null_limit[q > q_max - margin_to_limit] +=\
                qd_null_limit_max[q > q_max - margin_to_limit]
            qd_null_limit[q < q_min + margin_to_limit] +=\
                qd_null_limit_min[q < q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q

    def move_robot_start_pos(self):
        target_pos = START_POS
        target_vel = np.zeros(target_pos.shape)

        while np.sum(np.abs(target_pos - self.data.qpos[:7])) +\
              np.sum(np.abs(target_vel - self.data.qvel[:7])) > 1.0:
            action = self.controller.get_action(
                target_pos, target_vel, self.data.qpos[:7], self.data.qvel[:7]
            )
            action = 4 * np.clip(action, self.action_space.low, self.action_space.high)
            resultant_action = np.clip(
                action + self.data.qfrc_bias[:7].copy(),
                -q_torque_max,
                q_torque_max
            )

            self.render()
            try:
                self.do_simulation(resultant_action, self.frame_skip)
            except Exception as e:
                print(e)
                unstable_simulation = True

    def set_tcp_pos(self, desired_tcp_pos, hard_set=False):
        desired_tcp_pos[-1] = 0.05 if desired_tcp_pos[-1] < 0.05 else desired_tcp_pos[-1]
        init_robot_pos =  self.data.qpos[:7].copy()
        desired_tcp_quat = np.array([0, 1, 0, 0])
        robot_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)
        if hard_set:
            self.data.qpos[:7] = robot_joint_pos
        else:
            self.data.qpos[:7] = init_robot_pos
        self.data.qvel[:7] = np.zeros(robot_joint_pos.shape)
        return robot_joint_pos, self.robot_tcp_penalty(desired_tcp_pos[:2])

    def img_to_world(self, pixel_pos, cam="rgbd"):
        """
        Convert end-effector camera observation (rgb + depth) to simulation "world"
        coordinates using the camera rotation matrix and the intrinsic camera values.

        Args:
            pixel_pos (np.array): numpy array with, x and y image coordinates and depth
                value from depth camera
            cam (str): camera name
        Return:
            (np.array): x, y, z coordinated in the simulation
        """
        img_coords = np.append(pixel_pos[:2], 1.0)  # [u, v, 1]
        world_depth = self.depth_to_world_depth(img_coords, pixel_pos[-1])
        cam_pos = self.data.camera(cam).xpos

        world_point = np.linalg.lstsq(self.get_cam_mat(cam), img_coords, rcond=None)[0]
        world_point = -world_point[:3] / np.abs(world_point[-1])  # [X, Y, Z, 1]

        # line from camera to projected point
        direction = world_point - cam_pos
        world_coords = cam_pos + (world_depth / np.linalg.norm(direction)) * direction

        return world_coords

    def depth_to_world_depth(self, img_coords, depth, cam="rgbd"):
        """
        Undo zbuffer that maps real world distance to [0, 1]. Afterwards the real world
        depth is adjusted for camera distortion.

        Args:
            img_coords (np.array): pixel coordinates in the image to calculate distortion
                from the center
            depth (float): depth value from camera
            cam (str): camera name
        Return:
            (float): real "world" simulation distance from camera to pixel position
        """
        world_depth = self.near / (1 - depth * (1 - self.near / self.far))
        distortion = np.linalg.inv(self.focal_mat[cam][:3, :3]) @ img_coords
        depth_scale = (distortion / np.linalg.norm(distortion))[-1]
        world_depth /= depth_scale
        return world_depth

    def get_cam_mat(self, cam="rgbd"):
        """
        Args:
            cam (str): camera name
        Return:
            (np.array): camera matrix of shape (3, 4)
        """
        trans_mat, rot_mat = np.eye(4), np.eye(4)
        trans_mat[:3, 3] = -self.data.camera(cam).xpos
        rot_mat[:3, :3] = self.data.camera(cam).xmat.reshape((3, 3)).T
        return self.focal_mat[cam] @ rot_mat @ trans_mat


class BoxPushingBinSparse(BoxPushingBin):
    def __init__(
        self,
        num_boxes: int = 10,
        frame_skip: int = 10,
        width: int = 244,
        height: int = 244,
    ):
        self.bin_borders = np.random.rand(NUM_BINS, 6)  # 3 dims, each 2 border values
        super(BoxPushingBinSparse, self).__init__(num_boxes, frame_skip, width, height)
        bin_pos = [self.data.body("bin_" + str(b)).xpos[:3] for b in range(NUM_BINS)]
        bin_edges = np.array([BIN_SIZE, -BIN_SIZE] * 2 + [0, -2 * BIN_SIZE])
        self.bin_borders = np.array([np.repeat(p, 2) - bin_edges for p in bin_pos])

    def boxes_in_bin(self, box_pos):
        parallel_box_pos = np.repeat(np.expand_dims(box_pos, axis=1), NUM_BINS, axis=1)
        parallel_bin_borders = np.repeat(
            np.expand_dims(self.bin_borders, axis=0),
            len(box_pos),
            axis=0
        )
        bin_dist = np.concatenate(
            [
                parallel_box_pos - parallel_bin_borders[:,:,(0, 2, 4)],  # dist to low
                parallel_bin_borders[:,:,(1, 3, 5)] - parallel_box_pos  # dist to high
            ],
            axis=-1,
        )
        return np.where(np.sum(bin_dist > 0.0, axis=-1) >= 6)[0]

    def _get_reward(self, action, qpos, qvel, box_pos):
        """
        The sparse reward checks if a box is inside one of the bins. The condition is
        checked in parallel (no if statements). Only boxes that are inside the
        predetermined bin borders have distance > 0 on all 6 conditions. These boxes are
        then removed from the current trajectory. The instrinsic reward of smooth
        constrained movements is still applied from the base env reward.

        Args:
            action (np.array): action taken during step
            qpos (np.array): robot position, angle for each joint
            qvel (np.array): robot velocity, torque for each joint
            box_pos (np.array): (x, y, z) coordinated of all boxes, shape (num_boxes, 3)
        Return:
            (float): scalar reward value
        """
        penalty = super()._get_reward(action, qpos, qvel)
        boxes_in_bin = self.boxes_in_bin(np.array(box_pos)[self.boxes_out_bins])
        self.boxes_out_bins = np.delete(self.boxes_out_bins, boxes_in_bin)

        return penalty + len(boxes_in_bin) * 100


class BoxPushingBinDense(BoxPushingBinSparse):
    def __init__(
        self,
        num_boxes: int = 10,
        frame_skip: int = 10,
        width: int = 244,
        height: int = 244,
    ):
        super(BoxPushingBinDense, self).__init__(num_boxes, frame_skip, width, height)


    def _get_reward(self, action, qpos, qvel, box_pos):
        """
        The dense reward adds to the sparse reward the distance of each box to all of the
        bins. The distance across all bins and all boxes are summed up. The negative of
        value makes up the added reward which means the higher the distance the more
        negative reward. The distance used between the boxes and the bins is the
        mehalanobis distance across x and y axis, z is ignored since the boxes are
        assumed to be on the plane of the desk.

        Args:
            action (np.array): action taken during step
            qpos (np.array): robot position, angle for each joint
            qvel (np.array): robot velocity, torque for each joint
            box_pos (np.array): (x, y, z) coordinated of all boxes, shape (num_boxes, 3)
        Return:
            (float): scalar reward value
        """
        sparse_reward = super()._get_reward(action, qpos, qvel, box_pos)

        # Parallelize calculating mahalanobis distance by casting both box positions and
        # bin borders to (num_boxes, num_bins, 4 borders which are 2 for x and 2 for y)
        box_pos = np.repeat([b[:2] for b in box_pos], 2, axis=-1)
        parallel_box_pos = np.repeat(np.expand_dims(box_pos, axis=1), NUM_BINS, axis=1)
        parallel_bin_borders = np.repeat(
            np.expand_dims(self.bin_borders[:,:4], axis=0),
            len(box_pos),
            axis=0
        )
        dist_to_bin_borders = -np.sum(np.abs(parallel_box_pos - parallel_bin_borders))

        reward = sparse_reward + dist_to_bin_borders
        return reward
