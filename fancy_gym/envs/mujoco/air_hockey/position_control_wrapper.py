from collections import deque

import mujoco
import numpy as np
import scipy

from fancy_gym.envs.mujoco.air_hockey import seven_dof
from fancy_gym.envs.mujoco.air_hockey import three_dof


class PositionControl:
    def __init__(self, p_gain, d_gain, i_gain, interpolation_order=3, debug=False, *args, **kwargs):
        """
            Mixin that adds position controller to mujoco environments.

            Args:
                p_gain (float): Proportional controller gain
                d_gain (float): Differential controller gain
                i_gain (float): Integral controller gain
                interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
                    In the case of 2 agents it is a tuple, which describes the interpolation order for each agent
                debug (bool, True): If true it logs the controller performance into controller_record queue. The order of the
                    entries is desired_pos, current_pos, desired_vel, current_vel, desired_acc, jerk.
            """

        self.debug = debug

        super(PositionControl, self).__init__(*args, **kwargs)

        self.robot_model = self.env_info['robot']['robot_model']
        self.robot_data = self.env_info['robot']['robot_data']

        self.p_gain = np.array(p_gain * self.n_agents)
        self.d_gain = np.array(d_gain * self.n_agents)
        self.i_gain = np.array(i_gain * self.n_agents)

        self.prev_pos = np.zeros(len(self.actuator_joint_ids))
        self.prev_vel = np.zeros(len(self.actuator_joint_ids))
        self.prev_acc = np.zeros(len(self.actuator_joint_ids))
        self.i_error = np.zeros(len(self.actuator_joint_ids))
        self.prev_controller_cmd_pos = np.zeros(len(self.actuator_joint_ids))

        self.interp_order = interpolation_order if type(interpolation_order) is tuple else (interpolation_order,)

        self._num_env_joints = len(self.actuator_joint_ids)
        self.n_robot_joints = self.env_info['robot']["n_joints"]

        self.action_shape = [None] * self.n_agents

        for i in range(self.n_agents):
            if self.interp_order[i] is None:
                self.action_shape[i] = (int(self.dt / self._timestep), 3, self.n_robot_joints)
            elif self.interp_order[i] in [1, 2]:
                self.action_shape[i] = (self.n_robot_joints,)
            elif self.interp_order[i] in [3, 4, -1]:
                self.action_shape[i] = (2, self.n_robot_joints)
            elif self.interp_order[i] == 5:
                self.action_shape[i] = (3, self.n_robot_joints)

        self.traj = None

        self.jerk = np.zeros(self._num_env_joints)

        if self.debug:
            self.controller_record = deque(maxlen=self.info.horizon * self._n_intermediate_steps)

    def _enforce_safety_limits(self, desired_pos, desired_vel):
        # ROS safe controller
        pos = self.prev_controller_cmd_pos
        k = 20

        joint_pos_lim = np.tile(self.env_info['robot']['joint_pos_limit'], (1, self.n_agents))
        joint_vel_lim = np.tile(self.env_info['robot']['joint_vel_limit'], (1, self.n_agents))

        min_vel = np.minimum(np.maximum(-k * (pos - joint_pos_lim[0]), joint_vel_lim[0]), joint_vel_lim[1])

        max_vel = np.minimum(np.maximum(-k * (pos - joint_pos_lim[1]), joint_vel_lim[0]), joint_vel_lim[1])

        clipped_vel = np.minimum(np.maximum(desired_vel, min_vel), max_vel)

        min_pos = pos + min_vel * self._timestep
        max_pos = pos + max_vel * self._timestep

        clipped_pos = np.minimum(np.maximum(desired_pos, min_pos), max_pos)
        self.prev_controller_cmd_pos = clipped_pos.copy()

        return clipped_pos, clipped_vel

    def _controller(self, desired_pos, desired_vel, desired_acc, current_pos, current_vel):
        clipped_pos, clipped_vel = self._enforce_safety_limits(desired_pos, desired_vel)

        error = (clipped_pos - current_pos)

        self.i_error += self.i_gain * error * self._timestep
        torque = self.p_gain * error + self.d_gain * (clipped_vel - current_vel) + self.i_error

        # Acceleration FeedForward
        tau_ff = np.zeros(self.robot_model.nv)
        for i in range(self.n_agents):
            robot_joint_ids = np.arange(self.n_robot_joints) + self.n_robot_joints * i
            self.robot_data.qpos = current_pos[robot_joint_ids]
            self.robot_data.qvel = current_vel[robot_joint_ids]
            acc_ff = desired_acc[robot_joint_ids]
            mujoco.mj_forward(self.robot_model, self.robot_data)

            mujoco.mj_mulM(self.robot_model, self.robot_data, tau_ff, acc_ff)
            torque[robot_joint_ids] += tau_ff

            # Gravity Compensation and Coriolis and Centrifugal force
            torque[robot_joint_ids] += self.robot_data.qfrc_bias

            torque[robot_joint_ids] = np.minimum(np.maximum(torque[robot_joint_ids],
                                                            self.robot_model.actuator_ctrlrange[:, 0]),
                                                 self.robot_model.actuator_ctrlrange[:, 1])

        if self.debug:
            self.controller_record.append(
                np.concatenate([desired_pos, current_pos, desired_vel, current_vel, desired_acc, self.jerk]))

        return torque

    def _interpolate_trajectory(self, interp_order, action, i=0):
        tf = self.dt
        prev_pos = self.prev_pos[i*self.n_robot_joints:(i+1)*self.n_robot_joints]
        prev_vel = self.prev_vel[i*self.n_robot_joints:(i+1)*self.n_robot_joints]
        prev_acc = self.prev_acc[i*self.n_robot_joints:(i+1)*self.n_robot_joints]
        if interp_order == 1 and action.ndim == 1:
            coef = np.array([[1, 0], [1, tf]])
            results = np.vstack([prev_pos, action])
        elif interp_order == 2 and action.ndim == 1:
            coef = np.array([[1, 0, 0], [1, tf, tf ** 2], [0, 1, 0]])
            if np.linalg.norm(action - prev_pos) < 1e-3:
                prev_vel = np.zeros_like(prev_vel)
            results = np.vstack([prev_pos, action, prev_vel])
        elif interp_order == 3 and action.shape[0] == 2:
            coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
            results = np.vstack([prev_pos, action[0], prev_vel, action[1]])
        elif interp_order == 4 and action.shape[0] == 2:
            coef = np.array([[1, 0, 0, 0, 0], [1, tf, tf ** 2, tf ** 3, tf ** 4],
                             [0, 1, 0, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3],
                             [0, 0, 2, 0, 0]])
            results = np.vstack([prev_pos, action[0], prev_vel, action[1], prev_acc])
        elif interp_order == 5 and action.shape[0] == 3:
            coef = np.array([[1, 0, 0, 0, 0, 0], [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                             [0, 1, 0, 0, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                             [0, 0, 2, 0, 0, 0], [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])
            results = np.vstack([prev_pos, action[0], prev_vel, action[1], prev_acc, action[2]])
        elif interp_order == -1:
            # Interpolate position and velocity linearly
            pass
        else:
            raise ValueError("Undefined interpolator order or the action dimension does not match!")

        if interp_order > 0:
            A = scipy.linalg.block_diag(*[coef] * self.n_robot_joints)
            y = results.reshape(-2, order='F')
            weights = np.linalg.solve(A, y).reshape(self.n_robot_joints, interp_order + 1)
            weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
            weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)
        elif interp_order == -1:
            weights = np.vstack([prev_pos, (action[0] - prev_pos) / self.dt]).T
            weights_d = np.vstack([prev_vel, (action[1] - prev_vel) / self.dt]).T
            weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        if interp_order in [3, 4, 5]:
            self.jerk[i*self.n_robot_joints:(i+1)*self.n_robot_joints] = np.abs(weights_dd[:, 1]) + np.abs(weights_dd[:, 0] - prev_acc) / self._timestep
        else:
            self.jerk[i*self.n_robot_joints:(i+1)*self.n_robot_joints] = np.ones_like(prev_acc) * np.inf

        self.prev_pos[i*self.n_robot_joints:(i+1)*self.n_robot_joints] = np.polynomial.polynomial.polyval(tf, weights.T)
        self.prev_vel[i*self.n_robot_joints:(i+1)*self.n_robot_joints] = np.polynomial.polynomial.polyval(tf, weights_d.T)
        self.prev_acc[i*self.n_robot_joints:(i+1)*self.n_robot_joints] = np.polynomial.polynomial.polyval(tf, weights_dd.T)

        for t in np.linspace(self._timestep, self.dt, self._n_intermediate_steps):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            yield q, qd, qdd

    def reset(self, obs=None):
        obs = super(PositionControl, self).reset(obs)
        self.prev_pos = self._data.qpos[self.actuator_joint_ids]
        self.prev_vel = self._data.qvel[self.actuator_joint_ids]
        self.prev_acc = np.zeros(len(self.actuator_joint_ids))
        self.i_error = np.zeros(len(self.actuator_joint_ids))
        self.prev_controller_cmd_pos = self._data.qpos[self.actuator_joint_ids]

        if self.debug:
            self.controller_record = deque(maxlen=self.info.horizon * self._n_intermediate_steps)
        return obs

    def _step_init(self, obs, action):
        super(PositionControl, self)._step_init(obs, action)

        if self.n_agents == 1:
            self.traj = self._create_traj(self.interp_order[0], action)
        else:
            def _traj():
                traj_1 = self._create_traj(self.interp_order[0], action[0], 0)
                traj_2 = self._create_traj(self.interp_order[1], action[1], 1)

                for a1, a2 in zip(traj_1, traj_2):
                    yield np.hstack([a1, a2])

            self.traj = _traj()

    def _create_traj(self, interp_order, action, i=0):
        if interp_order is None:
            return iter(action)
        return self._interpolate_trajectory(interp_order, action, i)

    def _compute_action(self, obs, action):
        cur_pos, cur_vel = self.get_joints(obs)

        desired_pos, desired_vel, desired_acc = next(self.traj)

        return self._controller(desired_pos, desired_vel, desired_acc, cur_pos, cur_vel)

    def _preprocess_action(self, action):
        action = super(PositionControl, self)._preprocess_action(action)

        if self.n_agents == 1:
            assert action.shape == self.action_shape[0], f"Unexpected action shape. Expected {self.action_shape[0]} but got" \
                                                      f" {action.shape}"
        else:
            for i in range(self.n_agents):
                assert action[i].shape == self.action_shape[i], f"Unexpected action shape. Expected {self.action_shape[i]} but got" \
                                                          f" {action[i].shape}"

        return action


class PositionControlIIWA(PositionControl):
    def __init__(self, *args, **kwargs):
        p_gain = [1500., 1500., 1200., 1200., 1000., 1000., 500.]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        super(PositionControlIIWA, self).__init__(p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs)


class PositionControlPlanar(PositionControl):
    def __init__(self, *args, **kwargs):
        p_gain = [960, 480, 240]
        d_gain = [60, 20, 4]
        i_gain = [0, 0, 0]
        super(PositionControlPlanar, self).__init__(p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs)


class PlanarPositionHit(PositionControlPlanar, three_dof.AirHockeyHit):
    pass


class PlanarPositionDefend(PositionControlPlanar, three_dof.AirHockeyDefend):
    pass


class IiwaPositionHit(PositionControlIIWA, seven_dof.AirHockeyHit):
    pass

class IiwaPositionHitAirhocKIT2023(PositionControlIIWA, seven_dof.AirHockeyHitAirhocKIT2023):
    pass

class IiwaPositionDefend(PositionControlIIWA, seven_dof.AirHockeyDefend):
    pass

class IiwaPositionDefendAirhocKIT2023(PositionControlIIWA, seven_dof.AirHockeyDefendAirhocKIT2023):
    pass

class IiwaPositionTournament(PositionControlIIWA, seven_dof.AirHockeyTournament):
    pass
