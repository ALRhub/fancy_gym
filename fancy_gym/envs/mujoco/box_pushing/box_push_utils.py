import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from gym.envs.mujoco.mujoco_env import MujocoEnv

from scipy.interpolate import make_interp_spline

# joint constraints for Franka robot
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_min = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

q_dot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
q_dot_dot_max = np.array([90., 90., 90., 90., 12., 12., 12.])

desired_rod_quat = np.array([0.0, 1.0, 0.0, 0.0])


def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    :param curr_quat: current quaternion
    :param des_quat: desired quaternion
    :return: difference between current quaternion and desired quaternion
    """
    quatError = np.zeros((3, ))

    quatError[0] = (curr_quat[0] * des_quat[1] - des_quat[0] * curr_quat[1] -
                    curr_quat[3] * des_quat[2] + curr_quat[2] * des_quat[3])

    quatError[1] = (curr_quat[0] * des_quat[2] - des_quat[0] * curr_quat[2] +
                    curr_quat[3] * des_quat[1] - curr_quat[1] * des_quat[3])

    quatError[2] = (curr_quat[0] * des_quat[3] - des_quat[0] * curr_quat[3] -
                    curr_quat[2] * des_quat[1] + curr_quat[1] * des_quat[2])

    return quatError


def rotation_distance(p: np.array, q: np.array):
    """
    p: quaternion
    q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    product = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]
    theta = 2 * np.arccos(abs(product))
    return theta


def joint_limit_violate_penalty(joint_pos,
                                joint_vel,
                                enable_pos_limit=False,
                                enable_vel_limit=False):
    penalty = 0.
    p_coeff = 1.
    v_coeff = 1.
    # q_limit
    if enable_pos_limit:
        higher_indice = np.where(joint_pos > q_max)
        lower_indice = np.where(joint_pos < q_min)
        higher_error = joint_pos - q_max
        lower_error = q_min - joint_pos
        penalty -= p_coeff * (abs(np.sum(higher_error[higher_indice])) +
                              abs(np.sum(lower_error[lower_indice])))
    # q_dot_limit
    if enable_vel_limit:
        q_dot_error = abs(joint_vel) - abs(q_dot_max)
        q_dot_violate_idx = np.where(q_dot_error > 0.)
        penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_violate_idx]))
    return penalty


def rot_to_quat(theta, axis):
    quant = np.zeros(4)
    quant[0] = np.sin(theta / 2.)
    quant[1] = np.cos(theta / 2.) * axis[0]
    quant[2] = np.cos(theta / 2.) * axis[1]
    quant[3] = np.cos(theta / 2.) * axis[2]
    return quant


class PushBoxReward:
    def __init__(self):
        self.box_size = np.array([0.05, 0.05, 0.045])
        self.step_reward_joint_penalty = []
        self.step_reward_tcp = []
        self.step_reward_pos = []
        self.step_reward_rot = []
        self.step_reward_flip = []
        self.energy_cost = []

    def get_reward_trajectory(self):
        return np.array(self.step_reward_pos.copy()), np.array(
            self.step_reward_rot.copy())

    def reset(self):
        self.step_reward_pos = []
        self.step_reward_rot = []
        self.step_reward_tcp = []
        self.step_reward_joint_penalty = []
        self.step_reward_flip = []
        self.energy_cost = []

    def step_reward(self, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):

        joint_penalty = joint_limit_violate_penalty(qpos,
                                                    qvel,
                                                    enable_pos_limit=True,
                                                    enable_vel_limit=True)
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        energy_cost = -0.0005 * np.sum(np.square(action))

        reward = joint_penalty + tcp_box_dist_reward + \
                 box_goal_pos_dist_reward + box_goal_rot_dist_reward + energy_cost

        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        # self.step_reward_joint_penalty.append(joint_penalty)
        # self.step_reward_tcp.append(tcp_box_dist_reward)
        # self.step_reward_pos.append(box_goal_pos_dist_reward)
        # self.step_reward_rot.append(box_goal_rot_dist_reward)
        # self.energy_cost.append(energy_cost)
        return reward

    def sparse1_reward(self, episodic_end, box_pos, box_quat, target_pos,
                        target_quat, rod_tip_pos, rod_quat, qpos, qvel, action):

        reward = 0.
        joint_penalty = joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.0005 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episodic_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi * 100

        reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward

        return reward

    def sparse2_reward(self, episodic_end, box_pos, box_quat, target_pos,
                        target_quat, rod_tip_pos, rod_quat, qpos, qvel, action):

        reward = 0.
        joint_penalty = joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.0005 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episodic_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        if box_goal_dist < 0.1:
            reward += 300
            box_goal_pos_dist_reward = np.clip(- 3.5 * box_goal_dist * 100 * 3, -100, 0)
            box_goal_rot_dist_reward = np.clip(- rotation_distance(box_quat, target_quat)/np.pi * 100 * 1.5, -100, 0)
            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward

        return reward

    def plotRewards(self):
        length = np.array(self.step_reward_pos).shape[0]
        t = np.arange(0, length)
        fig, axs = plt.subplots(1, 1)
        axs.plot(t, self.step_reward_pos, label='pos')
        axs.plot(t, self.step_reward_rot, label='rot')
        axs.plot(t, self.step_reward_tcp, label='tcp')
        axs.plot(t,
                 np.array(self.step_reward_joint_penalty) / 0.5,
                 label='joint_penalty')
        # axs.plot(t, self.step_reward_flip, label='flip')
        axs.plot(t, self.energy_cost, label='energy_cost')
        plt.legend(loc='upper right')
        plt.xlabel('Steps')
        plt.title('Reward')
        plt.show()

class TrajectoryRecoder(object):
    def __init__(self, sim, prefix="episodic", max_length=1000):
        self.sim = sim
        self.prefix = prefix
        self.max_length = max_length
        self.reset()

    def update_sim(self, sim):
        self.sim = sim

    def reset(self):
        self.trajectory = []
        self.box_pos_dist_trajectory = []  # trajectory of box position distance to goal
        self.box_rot_dist_trajectory = []  # trajectory of box rotation distance to goal
        self.joint_pos_trajectory = []  # trajectory of joint position
        self.joint_vel_trajectory = []  # trajectory of joint velocity
        self.joint_torque_trajectory = []   # trajectory of joint torque
        self.length = 0

    def record(self):
        if self.sim is None:
            return

        self.joint_pos_trajectory.append(self.sim.data.qpos[:7].copy())
        self.joint_vel_trajectory.append(self.sim.data.qvel[:7].copy())
        self.joint_torque_trajectory.append(self.sim.data.qfrc_actuator[:7].copy())

        box_pos = self.sim.data.get_body_xpos("box_0")
        box_quat = self.sim.data.get_body_xquat("box_0")
        target_pos = self.sim.data.get_body_xpos("replan_target_pos")
        target_quat = self.sim.data.get_body_xquat("replan_target_pos")

        self.box_pos_dist_trajectory.append(np.linalg.norm(box_pos - target_pos))
        self.box_rot_dist_trajectory.append(rotation_distance(box_quat, target_quat))

        self.length += 1
        if self.length > self.max_length:
            self.joint_vel_trajectory.pop(0)
            self.joint_pos_trajectory.pop(0)
            self.joint_torque_trajectory.pop(0)
            self.box_pos_dist_trajectory.pop(0)
            self.box_rot_dist_trajectory.pop(0)
            self.length -= 1

    def get_trajectory(self):
        return self.trajectory

    def get_length(self):
        return self.length

    def plot_trajectories(self):
        self.plot_trajectory(self.joint_pos_trajectory,
                                   "joint_pos_trajectory")
        self.plot_trajectory(self.joint_vel_trajectory,
                                   "joint_vel_trajectory")
        self.plot_trajectory(self.joint_torque_trajectory,
                                   "joint_acc_trajectory")
        self.plot_trajectory(self.box_pos_dist_trajectory,
                                   "box_pos_dist_trajectory")
        self.plot_trajectory(self.box_rot_dist_trajectory,
                                   "box_rot_dist_trajectory")

    def plot_trajectory(self, trajectory, title: str):
        if len(trajectory) == 0:
            return
        trajectory = np.array(trajectory)
        length = trajectory.shape[0]
        t = np.arange(0, length)
        dim = trajectory.shape[1] if len(trajectory.shape) > 1 else 1
        fig, axs = plt.subplots(dim, 1, sharex=True)
        if dim == 1:
            axs.plot(t, trajectory)
        else:
            for i in range(dim):
                axs[i].plot(t, trajectory[:, i])
        # plt.legend(loc='upper right')
        plt.xlabel('Steps')
        plt.title(self.prefix + ": " + title)
        plt.show()

    def plot_box_trajectory(self):
        pass

    def save_trajectory(self, path: str):

        joint_pos_trajectory = np.array(self.joint_pos_trajectory)
        joint_vel_trajectory = np.array(self.joint_vel_trajectory)
        joint_torque_trajectory = np.array(self.joint_torque_trajectory)
        box_pos_dist_trajectory = np.array(self.box_pos_dist_trajectory)
        box_rot_dist_trajectory = np.array(self.box_rot_dist_trajectory)
        pd_dict = {}
        if joint_pos_trajectory.shape[0] > 0:
            for i in range(7):
                pd_dict["qpos_" + str(i)] = joint_pos_trajectory[:, i]
                pd_dict["qvel_" + str(i)] = joint_vel_trajectory[:, i]
                pd_dict["qfrc_" + str(i)] = joint_torque_trajectory[:, i]
            pd_dict["box_pos_dist"] = box_pos_dist_trajectory
            pd_dict["box_rot_dist"] = box_rot_dist_trajectory

            df = pd.DataFrame(pd_dict)
            folder_path = "/home/i53/student/hongyi_zhou/py_ws/alr_envs/alr_envs/alr/mujoco/box_pushing/recorded_trajectory/"
            save_path = folder_path + path + ".csv"
            df.to_csv(save_path)

def plot_trajectories(traj_dict:dict, traj_name:str):

    fig, axs = plt.subplots(1, 1, sharex=True)

    for key in traj_dict.keys():
        t = traj_dict[key][traj_name].shape[0]
        t = np.arange(0, t)
        spline = make_interp_spline(t, traj_dict[key][traj_name])
        t_ = np.linspace(t.min(), t.max(), 1000)
        y_ = spline(t_)
        axs.plot(t_, y_, label=key)

    plt.legend(loc='upper right')
    plt.title(traj_name)
    plt.xlabel('Steps')
    plt.show()


if __name__ == "__main__":

    ## sparse1 nodistcheck sparse2 withdistcheck
    folder_path = "/home/i53/student/hongyi_zhou/py_ws/alr_envs/alr_envs/alr/mujoco/box_pushing/recorded_trajectory/"
    exp_name = ["ppo_dense", "promp_dense_reward", "promp_sparse_nodistcheck", "promp_sparse_withdistcheck"]
    # exp_name = ["promp_sparse_nodistcheck"]
    trajectory_dict = { }
    for name in exp_name:
        data_path = folder_path + name + ".csv"
        df = pd.read_csv(data_path)
        trajectory_dict[name] = df

    # plot_trajectories(trajectory_dict, "box_pos_dist")
    # plot_trajectories(trajectory_dict, "box_rot_dist")
    plot_trajectories(trajectory_dict, "qpos_6")
    plot_trajectories(trajectory_dict, "qvel_6")
    plot_trajectories(trajectory_dict, "qfrc_6")
