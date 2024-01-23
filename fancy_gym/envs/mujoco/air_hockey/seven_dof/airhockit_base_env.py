import numpy as np
from gymnasium import spaces

from fancy_gym.envs.mujoco.air_hockey.seven_dof.env_single import AirHockeySingle
from fancy_gym.envs.mujoco.air_hockey.utils import inverse_kinematics, forward_kinematics, jacobian

class AirhocKIT2023BaseEnv(AirHockeySingle):
    def __init__(self, noise=False, **kwargs):
        super().__init__(**kwargs)
        obs_low = np.hstack([[-np.inf] * 37])
        obs_high = np.hstack([[np.inf] * 37])
        self.wrapper_obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
        self.wrapper_act_space = spaces.Box(low=np.repeat(-100., 6), high=np.repeat(100., 6))
        self.noise = noise

    # We don't need puck yaw observations
    def filter_obs(self, obs):
        obs = np.hstack([obs[0:2], obs[3:5], obs[6:12], obs[13:19], obs[20:]])
        return obs

    def add_noise(self, obs):
        if not self.noise:
            return
        obs[self.env_info["puck_pos_ids"]] += np.random.normal(0, 0.001, 3)
        obs[self.env_info["puck_vel_ids"]] += np.random.normal(0, 0.1, 3)
    
    def reset(self):
        self.last_acceleration = np.repeat(0., 6)
        obs = super().reset()
        self.add_noise(obs)
        self.interp_pos = obs[self.env_info["joint_pos_ids"]][:-1]
        self.interp_vel = obs[self.env_info["joint_vel_ids"]][:-1]

        self.last_planned_world_pos = self._fk(self.interp_pos)
        obs = np.hstack([
            obs, self.interp_pos, self.interp_vel, self.last_acceleration, self.last_planned_world_pos
        ])
        return self.filter_obs(obs)

    def step(self, action):
        action /= 10

        new_vel = self.interp_vel + action
        
        jerk = 2 * (new_vel - self.interp_vel - self.last_acceleration * 0.02) / (0.02 ** 2)
        new_pos = self.interp_pos + self.interp_vel * 0.02 + (1/2) * self.last_acceleration * (0.02 ** 2) + (1/6) * jerk * (0.02 ** 3)
        abs_action = np.vstack([np.hstack([new_pos, 0]), np.hstack([new_vel, 0])])

        self.interp_pos = new_pos
        self.interp_vel = new_vel
        self.last_acceleration += jerk * 0.02

        obs, rew, done, info = super().step(abs_action)
        self.add_noise(obs)
        self.last_planned_world_pos = self._fk(self.interp_pos)
        obs = np.hstack([
            obs, self.interp_pos, self.interp_vel, self.last_acceleration, self.last_planned_world_pos
        ])
        
        fatal_rew = self.check_fatal(obs)
        if fatal_rew != 0:
            return self.filter_obs(obs), fatal_rew, True, info
        
        return self.filter_obs(obs), rew, done, info
    
    def check_constraints(self, constraint_values):
        fatal_rew = 0

        j_pos_constr = constraint_values["joint_pos_constr"]
        if j_pos_constr.max() > 0:
            fatal_rew += j_pos_constr.max()

        j_vel_constr = constraint_values["joint_vel_constr"]
        if j_vel_constr.max() > 0:
            fatal_rew += j_vel_constr.max()

        ee_constr = constraint_values["ee_constr"]
        if ee_constr.max() > 0:
            fatal_rew += ee_constr.max()

        link_constr = constraint_values["link_constr"]
        if link_constr.max() > 0:
            fatal_rew += link_constr.max()
        
        return -fatal_rew

    def check_fatal(self, obs):
        fatal_rew = 0
        
        q = obs[self.env_info["joint_pos_ids"]]
        qd = obs[self.env_info["joint_vel_ids"]]
        constraint_values_obs = self.env_info["constraints"].fun(q, qd)
        fatal_rew += self.check_constraints(constraint_values_obs)

        return -fatal_rew

    def _fk(self, pos):
        res, _ = forward_kinematics(self.env_info["robot"]["robot_model"],
                                    self.env_info["robot"]["robot_data"], pos)
        return res.astype(np.float32)

    def _ik(self, world_pos, init_q=None):
        success, pos = inverse_kinematics(self.env_info["robot"]["robot_model"],
                                          self.env_info["robot"]["robot_data"],
                                          world_pos,
                                          initial_q=init_q)
        pos = pos.astype(np.float32)
        assert success
        return pos

    def _jacobian(self, pos):
        return jacobian(self.env_info["robot"]["robot_model"],
                        self.env_info["robot"]["robot_data"],
                        pos).astype(np.float32)
