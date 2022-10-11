import os
import time

import numpy as np
import mujoco_py
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
# from alr_envs.alr.mujoco.box_pushing.box_push_utils import PushBoxReward, rot_to_quat, q_max, q_min, get_quaternion_error
# from alr_envs.alr.mujoco.box_pushing.box_push_utils import q_dot_dot_max, rotation_distance
# from alr_envs.alr.mujoco.box_pushing.box_push_utils import TrajectoryRecoder
from fancy_gym.envs.mujoco.box_pushing.box_push_utils import PushBoxReward, rot_to_quat, q_max, q_min, get_quaternion_error
from fancy_gym.envs.mujoco.box_pushing.box_push_utils import q_dot_dot_max, rotation_distance
from fancy_gym.envs.mujoco.box_pushing.box_push_utils import TrajectoryRecoder

MAX_EPISODE_STEPS = 1000

INIT_BOX_POS_BOUND = np.array([[0.3, -0.45, -0.01], [0.6, 0.45, -0.01]])


class BOX_PUSH_Env_Gym(MujocoEnv, utils.EzPickle):
    def __init__(self, enable_gravity_comp=False, frame_skip=1, reward_type="Dense"):
        model_path = os.path.join(os.path.dirname(__file__), "assets", "box_pushing.xml")
        self.reward_type = reward_type
        assert reward_type in ["Dense", "Sparse1", "Sparse2"], "reward_type must be one of Dense, Sparse1, Sparse2"
        self.enable_gravity_comp = enable_gravity_comp
        self.frame_skip = frame_skip
        self.max_episode_steps = MAX_EPISODE_STEPS // self.frame_skip

        self.time_steps = 0
        self.init_qpos_box_push = np.array([
            0., 0., 0., -1.5, 0., 1.5, 0., 0., 0., 0.6, 0.45, 0.0, 1., 0., 0.,
            0.
        ])
        self.init_qvel_box_push = np.zeros(15)
        self._id_set = False
        self.reward = PushBoxReward()

        # utilities for IK
        self.J_reg = 1e-6
        self.W = np.diag([1, 1, 1, 1, 1, 1, 1])
        self.target_th_null = np.array([
            3.57795216e-09,
            1.74532920e-01,
            3.30500960e-08,
            -8.72664630e-01,
            -1.14096181e-07,
            1.22173047e00,
            7.85398126e-01,
        ])

        self.torque_bound_low = -q_dot_dot_max
        self.torque_bound_high = q_dot_dot_max

        self.episode_energy = 0.
        self.episode_end_pos_dist = 0.
        self.episode_end_rot_dist = 0.
        # self.trajectory_recorder = TrajectoryRecoder(None, max_length=100)
        # end of IK utilities
        super(BOX_PUSH_Env_Gym, self).__init__(model_path=model_path,
                                               frame_skip=self.frame_skip,
                                               mujoco_bindings="mujoco_py")
        utils.EzPickle.__init__(self)

        action_space_low = np.array([-1.0] * 7)
        action_space_high = np.array([1.0] * 7)
        self.action_space = spaces.Box(low=action_space_low,
                                       high=action_space_high,
                                       dtype='float32')
        # self.trajectory_recorder.update_sim(self.sim)

    def _set_ids(self):
        self.box_id = self.sim.model._body_name2id["box_0"]
        self.target_id = self.sim.model._body_name2id["target_pos"]
        self.rod_tip_site_id = self.sim.model.site_name2id("rod_tip")
        self._id_set = True

    def sample_context(self):
        # return np.random.uniform(INIT_BOX_POS_BOUND[0],
        #                          INIT_BOX_POS_BOUND[1],
        #                          size=INIT_BOX_POS_BOUND[0].shape)
        # pos = np.random.uniform(INIT_BOX_POS_BOUND[0],
        #                         INIT_BOX_POS_BOUND[1],
        #                         size=INIT_BOX_POS_BOUND[0].shape)
        pos = np.array([0.4, 0.3, -0.01]) # was 0.45 0.4
        # theta = np.random.uniform(0, np.pi * 2)
        theta = 0.0
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        return np.concatenate((pos, quat))

    def generate_specified_context(self, goal_x, goal_y, goal_rot):
        pos = np.array([goal_x, goal_y, -0.01])
        quat = rot_to_quat(goal_rot, np.array([0, 0, 1]))
        return np.concatenate((pos, quat))

    def reset_model(self):
        self.reward.reset()

        self.episode_energy = 0.
        self.episode_end_pos_dist = 0.
        self.episode_end_rot_dist = 0.

        self.set_state(self.init_qpos_box_push, self.init_qvel_box_push)
        box_init_pos = self.sample_context()
        box_init_pos[0] = 0.4
        box_init_pos[1] = 0.3
        box_init_pos[-4:] = np.array([0, 0, 0, 1.])

        box_target_pos = self.sample_context()

        # if both box and target are in the same position, sample again
        # while np.linalg.norm(box_init_pos[:3] - box_target_pos[:3]) < 0.3:
        #     box_target_pos = self.sample_context()
        box_target_pos[0] = 0.4  # was 0.4
        box_target_pos[1] = -0.3  # was -0.3
        # box_target_pos = self.generate_specified_context(0.45, -0.25, np.pi)

        self.sim.model.body_pos[2] = box_target_pos[:3]
        self.sim.model.body_quat[2] = box_target_pos[-4:]

        self.sim.model.body_pos[3] = box_target_pos[:3]
        self.sim.model.body_quat[3] = box_target_pos[-4:]

        desired_ee_pos = box_init_pos[:3].copy()
        desired_ee_pos[2] += 0.15
        desired_ee_quat = np.array([0, 1, 0, 0])
        desired_joint_pos = self.findJointPosition(desired_ee_pos,
                                                   desired_ee_quat)
        self.sim.data.qpos[:7] = desired_joint_pos
        self.sim.data.set_joint_qpos('box_joint', box_init_pos)
        # self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        self.sim.forward()
        self.time_steps = 0

        # reset trajectory recorder
        # self.trajectory_recorder.plot_trajectories()
        # self.trajectory_recorder.save_trajectory("ppo_dense")
        # self.trajectory_recorder.reset()

        return self._get_obs()

    def _get_obs(self):
        box_pos = self.sim.data.get_body_xpos("box_0")
        box_quat = self.sim.data.get_body_xquat("box_0")
        target_pos = self.sim.data.get_body_xpos("replan_target_pos")
        target_quat = self.sim.data.get_body_xquat("replan_target_pos")
        rod_tip_pos = self.sim.data.site_xpos[self.rod_tip_site_id]
        rod_quat = self.sim.data.get_body_xquat("push_rod")
        obs = np.concatenate([
            self.sim.data.qpos[:7].copy(),
            self.sim.data.qvel[:7].copy(),
            self.sim.data.qfrc_bias[:7].copy(),
            # self.sim.data.qfrc_actuator[:7].copy(),
            rod_tip_pos,
            rod_quat,
            box_pos,
            box_quat,
            target_pos,
            target_quat
            # np.array([self.time_steps / 100.])
        ])
        return obs

    def step(self, action):
        if not self._id_set:
            self._set_ids()
        invalid_flag = False
        done = False
        self.time_steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.enable_gravity_comp:
            action = action * 10.  # rescale action
            resultant_action = action + self.sim.data.qfrc_bias[:7].copy()
        else:
            resultant_action = action * 10.

        resultant_action = np.clip(resultant_action, self.torque_bound_low,
                                   self.torque_bound_high)

        # the replan_target_pos was align with the target_pos before 800
        # if self.time_steps == 20:
        #     new_target_pos = np.array([1., 1., -0.01])
        #     new_target_quat = np.array([0, 1, 0, 0])
        #     while new_target_pos[0] < 0.4 or new_target_pos[0] > 0.6 or abs(
        #             new_target_pos[1]) > 0.45:
        #         pos_change = np.random.uniform(-0.25, 0.2, 3)
        #         pos_change[-1] = 0.
        #         # pos_change[-2] = 0.
        #         # self.sim.model.body_pos[3] = self.sim.data.get_body_xpos("target_pos") + pos_change
        #         # self.sim.model.body_quat[3] = self.sim.data.get_body_xquat("target_pos")
        #         new_target_pos = self.sim.data.get_body_xpos(
        #             "target_pos") + pos_change
        #         # new_target_quat = self.sim.data.get_body_xquat("target_pos")
        #     old_target_quat = self.sim.data.get_body_xquat("target_pos")
        #     new_target_quat = None
        #     while new_target_quat is None or rotation_distance(
        #             new_target_quat, old_target_quat) > np.pi / 2.:
        #         theta = np.random.uniform(0, np.pi * 2)
        #         new_target_quat = rot_to_quat(theta, np.array([0, 0, 1]))
        #
        #     self.sim.model.body_pos[3] = new_target_pos
        #     self.sim.model.body_quat[3] = new_target_quat
        #     self.sim.forward()

        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except mujoco_py.MujocoException as e:
            print(e)
            invalid_flag = True

        # record the trajectory
        # if self.time_steps % (10//self.frame_skip) == 0:
        # self.trajectory_recorder.record()

        box_pos = self.sim.data.get_body_xpos("box_0")
        box_quat = self.sim.data.get_body_xquat("box_0")
        target_pos = self.sim.data.get_body_xpos("replan_target_pos")
        target_quat = self.sim.data.get_body_xquat("replan_target_pos")
        rod_tip_pos = self.sim.data.site_xpos[self.rod_tip_site_id].copy()
        rod_quat = self.sim.data.get_body_xquat("push_rod")
        qpos = self.sim.data.qpos[:7].copy()
        qvel = self.sim.data.qvel[:7].copy()


        episode_end = False
        self.episode_energy += np.sum(np.square(action))
        if self.time_steps >= 100 - 1:
            episode_end = True
            done = True
            self.episode_end_pos_dist = np.linalg.norm(box_pos - target_pos)
            self.episode_end_rot_dist = rotation_distance(
                box_quat, target_quat)

        if self.reward_type == "Dense":
            reward = self.reward.step_reward(box_pos, box_quat, target_pos,
                                            target_quat, rod_tip_pos, rod_quat,
                                            qpos, qvel, action)
        elif self.reward_type == "Sparse1":
            reward = self.reward.sparse1_reward(episode_end, box_pos, box_quat,
                                                target_pos, target_quat,
                                                rod_tip_pos, rod_quat, qpos, qvel,
                                                action)
        elif self.reward_type == "Sparse2":
            reward = self.reward.sparse2_reward(episode_end, box_pos, box_quat,
                                                target_pos, target_quat, rod_tip_pos,
                                                rod_quat, qpos, qvel, action)
        else:
            raise NotImplementedError("Unknown reward type: {}".format(
                self.reward_type))

        if invalid_flag:
            reward = -25

        obs = self._get_obs()
        infos = {
            'episode_end':
            episode_end,
            'box_goal_pos_dist':
            self.episode_end_pos_dist,
            'box_goal_rot_dist':
            self.episode_end_rot_dist,
            'episode_energy':
            self.episode_energy if episode_end else 0.,
            'is_success':
            True if episode_end and self.episode_end_pos_dist < 0.05
            and self.episode_end_rot_dist < 0.5 else False,
            'num_steps':
            self.time_steps
        }
        return obs, reward, done, infos

    def getJacobian(self, body_id="tcp", q=np.zeros(7)):

        self.sim.data.qpos[:7] = q
        self.sim.forward()
        jacp = self.sim.data.get_body_jacp(body_id).reshape(3, -1)[:, :7]
        jacr = self.sim.data.get_body_jacr(body_id).reshape(3, -1)[:, :7]
        jac = np.concatenate((jacp, jacr), axis=0)
        return jac

    def findJointPosition(self, desiredPos=None, desiredQuat=None):

        eps = 1e-5
        IT_MAX = 1000
        DT = 1e-3

        i = 0
        self.pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        self.pgain_null = 5 * np.array([
            7.675519770796831,
            2.676935478437176,
            8.539040163444975,
            1.270446361314313,
            8.87752182480855,
            2.186782233762969,
            4.414432577659688,
        ])
        self.pgain_limit = 20

        q = self.sim.data.qpos[:7].copy()
        qd_d = np.zeros(q.shape)
        oldErrNorm = np.inf

        # if (desiredPos is None):
        #     desiredPos = self.desiredTaskPosition[:3]
        #
        # if (desiredQuat is None):
        #     desiredQuat = self.desiredTaskPosition[3:]

        while True:
            oldQ = q
            q = q + DT * qd_d

            q = np.clip(q, q_min, q_max)

            # cartPos, orient = self.sim.data.site_xpos[self.rod_tip_site_id]
            cartPos = self.sim.data.get_body_xpos("tcp")
            orient = self.sim.data.get_body_xquat("tcp")
            cpos_err = desiredPos - cartPos

            if np.linalg.norm(orient -
                              desiredQuat) > np.linalg.norm(orient +
                                                            desiredQuat):
                orient = -orient

            cpos_err = np.clip(cpos_err, -0.1, 0.1)
            cquat_err = np.clip(
                get_quaternion_error(orient, desiredQuat),
                -0.5,
                0.5,
            )
            err = np.hstack((cpos_err, cquat_err))

            errNorm = np.sum(cpos_err**2) + np.sum((orient - desiredQuat)**2)

            if errNorm > oldErrNorm:
                q = oldQ
                DT = DT * 0.7
                continue
            else:
                DT = DT * 1.025

            if errNorm < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            # if not i % 1:
            #print('%d: error = %s, %s, %s' % (i, errNorm, oldErrNorm, DT))

            oldErrNorm = errNorm

            J = self.getJacobian(q=q)

            Jw = J.dot(self.W)

            # J *  W * J' + reg * I
            JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(J.shape[0])

            # Null space movement
            qd_null = self.pgain_null * (self.target_th_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = self.pgain_limit * (q_max - margin_to_limit -
                                                    q)
            qd_null_limit_min = self.pgain_limit * (q_min + margin_to_limit -
                                                    q)
            qd_null_limit[q > q_max - margin_to_limit] += qd_null_limit_max[
                q > q_max - margin_to_limit]
            qd_null_limit[q < q_min + margin_to_limit] += qd_null_limit_min[
                q < q_min + margin_to_limit]

            qd_null += qd_null_limit
            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, self.pgain * err - J.dot(qd_null))
            # qd_d = self.pgain * err
            qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        # print("Final IK error (%d iterations):  %s" % (i, errNorm))
        return q


if __name__ == "__main__":
    env = BOX_PUSH_Env_Gym(enable_gravity_comp=True, frame_skip=10)
    env.reset()
    for j in range(60):
        old_obs = env.reset()
        done = False
        for _ in range(100):
            env.render(mode='human')
            # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # action = np.array([0.0] * 7)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            qpos = env.sim.data.qpos
            # qpos[:7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            # qpos[:7] = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
            # qvel = env.sim.data.qvel
            # qvel[:7] = [0, 0, 0, 0, 0, 0, 0]
            # env.set_state(qpos, qvel)
            # print("diff between old and new obs: ", np.linalg.norm(obs - old_obs))
            old_obs = obs
    print("========================================")
