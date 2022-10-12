import numpy as np


# joint constraints for Franka robot
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

q_dot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
q_torque_max = np.array([90., 90., 90., 90., 12., 12., 12.])
#
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


def rot_to_quat(theta, axis):
    quant = np.zeros(4)
    quant[0] = np.sin(theta / 2.)
    quant[1] = np.cos(theta / 2.) * axis[0]
    quant[2] = np.cos(theta / 2.) * axis[1]
    quant[3] = np.cos(theta / 2.) * axis[2]
    return quant



class RewardBase():
    def __init__(self, q_max, q_min, q_dot_max):
        self._reward = 0.
        self._done = False
        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max

    def get_reward(self, episodic_end, box_pos, box_quat, target_pos, target_quat,
                   rod_tip_pos, rod_quat, qpos, qvel, action):
        raise NotImplementedError

    def _joint_limit_violate_penalty(self,
                                    qpos,
                                    qvel,
                                    enable_pos_limit=False,
                                    enable_vel_limit=False):
        penalty = 0.
        p_coeff = 1.
        v_coeff = 1.
        # q_limit
        if enable_pos_limit:
            higher_indice = np.where(qpos > self._q_max)
            lower_indice = np.where(qpos < self._q_min)
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (abs(np.sum(higher_error[higher_indice])) +
                                  abs(np.sum(lower_error[lower_indice])))
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            q_dot_violate_idx = np.where(q_dot_error > 0.)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_violate_idx]))
        return penalty




class DenseReward(RewardBase):
    def __init__(self, q_max, q_min, q_dot_max):
        super(DenseReward, self).__init__(q_max, q_min, q_dot_max)

    def get_reward(self, episodic_end, box_pos, box_quat, target_pos, target_quat,
                   rod_tip_pos, rod_quat, qpos, qvel, action):
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

        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        return reward




class TemporalSparseReward(RewardBase):
    def __init__(self, q_max, q_min, q_dot_max):
        super(TemporalSparseReward, self).__init__(q_max, q_min, q_dot_max)

    def get_reward(self, episodic_end, box_pos, box_quat, target_pos, target_quat,
                   rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
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




class TemporalSpatialSparseReward(RewardBase):
    def __init__(self, q_max, q_min, q_dot_max):
        super(TemporalSpatialSparseReward, self).__init__(q_max, q_min, q_dot_max)

    def get_reward(self, episodic_end, box_pos, box_quat, target_pos, target_quat,
                   rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
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



def BoxPushingReward(reward_type, q_max, q_min, q_dot_max):
    if reward_type == 'Dense':
        return DenseReward(q_max, q_min, q_dot_max)
    elif reward_type == 'TemporalSparse':
        return TemporalSparseReward(q_max, q_min, q_dot_max)
    elif reward_type == 'TemporalSpatialSparse':
        return TemporalSpatialSparseReward(q_max, q_min, q_dot_max)
    else:
        raise NotImplementedError