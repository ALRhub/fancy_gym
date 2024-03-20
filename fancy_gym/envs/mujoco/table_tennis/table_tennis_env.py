import os

import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv

from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import is_init_state_valid, magnus_force
from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import jnt_pos_low, jnt_pos_high, jnt_vel_low, jnt_vel_high

import mujoco

MAX_EPISODE_STEPS_TABLE_TENNIS = 350
MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER = 300

CONTEXT_BOUNDS_2DIMS = np.array([[-1.0, -0.65], [-0.2, 0.65]])
CONTEXT_BOUNDS_4DIMS = np.array([[-1.0, -0.65, -1.0, -0.65],
                                 [-0.2, 0.65, -0.2, 0.65]])
CONTEXT_BOUNDS_SWICHING = np.array([[-1.0, -0.65, -1.0, 0.],
                                    [-0.2, 0.65, -0.2, 0.65]])


DEFAULT_ROBOT_INIT_POS = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5])
DEFAULT_ROBOT_INIT_VEL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class TableTennisEnv(MujocoEnv, utils.EzPickle):
    """
    7 DoF table tennis environment
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125
    }

    def __init__(self, ctxt_dim: int = 4, frame_skip: int = 4,
                 goal_switching_step: int = None,
                 enable_artificial_wind: bool = False,
                 random_pos_scale: float = 0.0,
                 random_vel_scale: float = 0.0,
                 **kwargs,
                ):
        utils.EzPickle.__init__(**locals())
        self._steps = 0

        self._hit_ball = False
        self._ball_land_on_table = False
        self._ball_contact_after_hit = False
        self._ball_return_success = False
        self._ball_landing_pos = None
        self._init_ball_state = None
        self._terminated = False

        self._id_set = False

        # initial robot state
        self._random_pos_scale = random_pos_scale
        self._random_vel_scale = random_vel_scale

        # reward calculation
        self.ball_landing_pos = None
        self._goal_pos = np.zeros(2)
        self._ball_traj = []
        self._racket_traj = []

        self._goal_switching_step = goal_switching_step

        self._enable_artificial_wind = enable_artificial_wind

        self._artificial_force = 0.

        if not hasattr(self, 'observation_space'):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
            )

        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "xml", "table_tennis_env.xml"),
                           frame_skip=frame_skip,
                           observation_space=self.observation_space,
                           **kwargs)

        self.render_active = False

        if ctxt_dim == 2:
            self.context_bounds = CONTEXT_BOUNDS_2DIMS
        elif ctxt_dim == 4:
            self.context_bounds = CONTEXT_BOUNDS_4DIMS
            if self._goal_switching_step is not None:
                self.context_bounds = CONTEXT_BOUNDS_SWICHING
        else:
            raise NotImplementedError

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        self._wind_vel = np.zeros(3)

    def _set_ids(self):
        self._floor_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._ball_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_ball_contact")
        self._bat_front_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bat")
        self._bat_back_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bat_back")
        self._table_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_table")
        self._id_set = True

    def step(self, action):
        if not self._id_set:
            self._set_ids()

        unstable_simulation = False

        if self._steps == self._goal_switching_step and self.np_random.uniform() < 0.5:
            new_goal_pos = self._generate_goal_pos(random=True)
            new_goal_pos[1] = -new_goal_pos[1]
            self._goal_pos = new_goal_pos
            self.model.body_pos[5] = np.concatenate([self._goal_pos, [0.77]])
            mujoco.mj_forward(self.model, self.data)

        for _ in range(self.frame_skip):
            if self._enable_artificial_wind:
                self.data.qfrc_applied[-2] = self._artificial_force
            try:
                self.do_simulation(action, 1)
            except Exception as e:
                print("Simulation get unstable return with MujocoException: ", e)
                unstable_simulation = True
                self._terminated = True
                break

            if not self._hit_ball:
                self._hit_ball = self._contact_checker(self._ball_contact_id, self._bat_front_id) or \
                    self._contact_checker(self._ball_contact_id, self._bat_back_id)
                if not self._hit_ball:
                    ball_land_on_floor_no_hit = self._contact_checker(self._ball_contact_id, self._floor_contact_id)
                    if ball_land_on_floor_no_hit:
                        self._ball_landing_pos = self.data.body("target_ball").xpos.copy()
                        self._terminated = True
            if self._hit_ball and not self._ball_contact_after_hit:
                if self._contact_checker(self._ball_contact_id, self._floor_contact_id):  # first check contact with floor
                    self._ball_contact_after_hit = True
                    self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                    self._terminated = True
                elif self._contact_checker(self._ball_contact_id, self._table_contact_id):  # second check contact with table
                    self._ball_contact_after_hit = True
                    self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                    if self._ball_landing_pos[0] < 0.:  # ball lands on the opponent side
                        self._ball_return_success = True
                    self._terminated = True

            # update ball trajectory & racket trajectory
            self._ball_traj.append(self.data.body("target_ball").xpos.copy())
            self._racket_traj.append(self.data.geom("bat").xpos.copy())

        self._steps += 1
        self._terminated = True if self._steps >= MAX_EPISODE_STEPS_TABLE_TENNIS else self._terminated

        reward = -25 if unstable_simulation else self._get_reward(self._terminated)

        land_dist_err = np.linalg.norm(self._ball_landing_pos[:-1] - self._goal_pos) \
            if self._ball_landing_pos is not None else 10.

        info = {
            "hit_ball": self._hit_ball,
            "ball_returned_success": self._ball_return_success,
            "land_dist_error": land_dist_err,
            "is_success": self._ball_return_success and land_dist_err < 0.2,
            "num_steps": self._steps,
        }

        terminated, truncated = self._terminated, self._steps == MAX_EPISODE_STEPS_TABLE_TENNIS

        if self.render_active and self.render_mode=='human':
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        self.render_active = True
        return super().render()

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]
            if (con.geom1 == id_1 and con.geom2 == id_2) or (con.geom1 == id_2 and con.geom2 == id_1):
                return True
        return False

    def get_initial_robot_state(self):

        robot_init_pos = DEFAULT_ROBOT_INIT_POS + \
                         self.np_random.uniform(-1.0, 1.0, size=7) *\
                         np.array([5.2, 4.0, 5.6, 4.0, 6.1, 3.2, 4.4]) *\
                         self._random_pos_scale

        robot_init_vel = DEFAULT_ROBOT_INIT_VEL + self.np_random.uniform(-1.0, 1.0, size=7) * self._random_vel_scale

        return np.clip(robot_init_pos, jnt_pos_low, jnt_pos_high), np.clip(robot_init_vel, jnt_vel_low, jnt_vel_high)

    def reset_model(self):
        self._steps = 0
        self._init_ball_state = self._generate_valid_init_ball(random_pos=True, random_vel=False)
        self._goal_pos = self._generate_goal_pos(random=True)
        self.data.joint("tar_x").qpos = self._init_ball_state[0]
        self.data.joint("tar_y").qpos = self._init_ball_state[1]
        self.data.joint("tar_z").qpos = self._init_ball_state[2]
        self.data.joint("tar_x").qvel = self._init_ball_state[3]
        self.data.joint("tar_y").qvel = self._init_ball_state[4]
        self.data.joint("tar_z").qvel = self._init_ball_state[5]

        if self._enable_artificial_wind:
            self._artificial_force = self.np_random.uniform(low=-0.1, high=0.1)

        self.model.body_pos[5] = np.concatenate([self._goal_pos, [0.77]])

        robot_init_pos, robot_init_vel = self.get_initial_robot_state()

        self.data.qpos[:7] = robot_init_pos
        self.data.qvel[:7] = robot_init_vel

        mujoco.mj_forward(self.model, self.data)

        self._hit_ball = False
        self._ball_land_on_table = False
        self._ball_contact_after_hit = False
        self._ball_return_success = False
        self._ball_landing_pos = None
        self._terminated = False
        self._ball_traj = []
        self._racket_traj = []
        return self._get_obs()

    def _generate_goal_pos(self, random=True):
        if random:
            return self.np_random.uniform(low=self.context_bounds[0][-2:], high=self.context_bounds[1][-2:])
        else:
            return np.array([-0.6, 0.4])

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:7].copy(),
            self.data.qvel.flat[:7].copy(),
            self.data.joint("tar_x").qpos.copy(),
            self.data.joint("tar_y").qpos.copy(),
            self.data.joint("tar_z").qpos.copy(),
            self._goal_pos.copy(),
        ])
        return obs

    def _get_reward(self, terminated):
        if not terminated:
            return 0
        min_r_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._racket_traj), axis=1))
        if not self._hit_ball:
            return 0.2 * (1 - np.tanh(min_r_b_dist**2))
        if self._ball_landing_pos is None:
            min_b_des_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj)[:, :2] - self._goal_pos[:2], axis=1))
            return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + (1 - np.tanh(min_b_des_b_dist**2))
        min_b_des_b_land_dist = np.linalg.norm(self._goal_pos[:2] - self._ball_landing_pos[:2])
        over_net_bonus = int(self._ball_landing_pos[0] < 0)
        return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + 4 * (1 - np.tanh(min_b_des_b_land_dist ** 2)) + over_net_bonus

    def _generate_random_ball(self, random_pos=False, random_vel=False):
        x_pos, y_pos, z_pos = -0.5, 0.35, 1.75
        x_vel, y_vel, z_vel = 2.5, 0., 0.5
        if random_pos:
            x_pos = self.np_random.uniform(low=self.context_bounds[0][0], high=self.context_bounds[1][0])
            y_pos = self.np_random.uniform(low=self.context_bounds[0][1], high=self.context_bounds[1][1])
        if random_vel:
            x_vel = self.np_random.uniform(low=2.0, high=3.0)
        init_ball_state = np.array([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel])
        return init_ball_state

    def _generate_valid_init_ball(self, random_pos=False, random_vel=False):
        init_ball_state = self._generate_random_ball(random_pos=random_pos, random_vel=random_vel)
        while not is_init_state_valid(init_ball_state):
            init_ball_state = self._generate_random_ball(random_pos=random_pos, random_vel=random_vel)
        return init_ball_state

    def _get_traj_invalid_penalty(self, action, pos_traj, tau_bound, delay_bound):
        tau_invalid_penalty = 3 * (np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]]))
        delay_invalid_penalty = 3 * (np.max([0, action[1] - delay_bound[1]]) + np.max([0, delay_bound[0] - action[1]]))
        violate_high_bound_error = np.mean(np.maximum(pos_traj - jnt_pos_high, 0))
        violate_low_bound_error = np.mean(np.maximum(jnt_pos_low - pos_traj, 0))
        invalid_penalty = tau_invalid_penalty + delay_invalid_penalty + \
            violate_high_bound_error + violate_low_bound_error
        return -invalid_penalty

    def get_invalid_traj_step_return(self, action, pos_traj, contextual_obs, tau_bound, delay_bound):
        obs = self._get_obs() if contextual_obs else np.concatenate([self._get_obs(), np.array([0])])  # 0 for invalid traj
        penalty = self._get_traj_invalid_penalty(action, pos_traj, tau_bound, delay_bound)
        return obs, penalty, False, True, {
            "hit_ball": [False],
            "ball_returned_success": [False],
            "land_dist_error": [10.],
            "is_success": [False],
            "trajectory_length": 1,
            "num_steps": [1],
        }

    @staticmethod
    def check_traj_validity(action, pos_traj, vel_traj, tau_bound, delay_bound):
        time_invalid = action[0] > tau_bound[1] or action[0] < tau_bound[0] \
            or action[1] > delay_bound[1] or action[1] < delay_bound[0]
        if time_invalid or np.any(pos_traj > jnt_pos_high) or np.any(pos_traj < jnt_pos_low):
            return False, pos_traj, vel_traj
        return True, pos_traj, vel_traj

class TableTennisMarkov(TableTennisEnv):
    def _get_reward2(self, hit_now, land_now):

        # Phase 1 not hit ball
        if not self._hit_ball:
            # Not hit ball
            min_r_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._racket_traj), axis=1))
            return 0.005 * (1 - np.tanh(min_r_b_dist**2))

        # Phase 2 hit ball now
        elif self._hit_ball and hit_now:
            return 2

        # Phase 3 hit ball already and not land yet
        elif self._hit_ball and self._ball_landing_pos is None:
            min_b_des_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj)[:,:2] - self._goal_pos[:2], axis=1))
            return 0.02 * (1 - np.tanh(min_b_des_b_dist**2))

        # Phase 4 hit ball already and land now
        elif self._hit_ball and land_now:
            over_net_bonus = int(self._ball_landing_pos[0] < 0)
            min_b_des_b_land_dist = np.linalg.norm(self._goal_pos[:2] - self._ball_landing_pos[:2])
            return 4 * (1 - np.tanh(min_b_des_b_land_dist ** 2)) + over_net_bonus

        # Phase 5 hit ball already and land already
        elif self._hit_ball and not land_now and self._ball_landing_pos is not None:
            return 0

        else:
            raise NotImplementedError

    def _get_reward(self, terminated):
        # if not terminated:
        #     return 0

        min_r_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._racket_traj), axis=1))
        if not self._hit_ball:
            # Not hit ball
            return 0.2 * (1 - np.tanh(min_r_b_dist**2))
        elif self._ball_landing_pos is None:
            # Hit ball but not landing pos
            min_b_des_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj)[:,:2] - self._goal_pos[:2], axis=1))
            return 2 + (1 - np.tanh(min_b_des_b_dist**2))
        else:
            # Hit ball and land
            min_b_des_b_land_dist = np.linalg.norm(self._goal_pos[:2] - self._ball_landing_pos[:2])
            over_net_bonus = int(self._ball_landing_pos[0] < 0)
            return 2 + 4 * (1 - np.tanh(min_b_des_b_land_dist ** 2)) + over_net_bonus


    def _get_traj_invalid_penalty(self, action, pos_traj, tau_bound, delay_bound):
        tau_invalid_penalty = 3 * (np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]]))
        delay_invalid_penalty = 3 * (np.max([0, action[1] - delay_bound[1]]) + np.max([0, delay_bound[0] - action[1]]))
        violate_high_bound_error = np.mean(np.maximum(pos_traj - jnt_pos_high, 0))
        violate_low_bound_error = np.mean(np.maximum(jnt_pos_low - pos_traj, 0))
        invalid_penalty = tau_invalid_penalty + delay_invalid_penalty + \
                          violate_high_bound_error + violate_low_bound_error
        return -invalid_penalty

    def get_invalid_traj_step_penalty(self, pos_traj):
        violate_high_bound_error = (
            np.maximum(pos_traj - jnt_pos_high, 0).mean())
        violate_low_bound_error = (
            np.maximum(jnt_pos_low - pos_traj, 0).mean())
        invalid_penalty = violate_high_bound_error + violate_low_bound_error


    def _update_game_state(self, action):
        for _ in range(self.frame_skip):
            if self._enable_artificial_wind:
                self.data.qfrc_applied[-2] = self._artificial_force
            try:
                self.do_simulation(action, 1)
            except Exception as e:
                print("Simulation get unstable return with MujocoException: ", e)
                unstable_simulation = True
                self._terminated = True
                break

            # Update game state
            if not self._terminated:
                if not self._hit_ball:
                    self._hit_ball = self._contact_checker(self._ball_contact_id, self._bat_front_id) or \
                                    self._contact_checker(self._ball_contact_id, self._bat_back_id)
                    if not self._hit_ball:
                        ball_land_on_floor_no_hit = self._contact_checker(self._ball_contact_id, self._floor_contact_id)
                        if ball_land_on_floor_no_hit:
                            self._ball_landing_pos = self.data.body("target_ball").xpos.copy()
                            self._terminated = True
                if self._hit_ball and not self._ball_contact_after_hit:
                    if self._contact_checker(self._ball_contact_id, self._floor_contact_id):  # first check contact with floor
                        self._ball_contact_after_hit = True
                        self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                        self._terminated = True
                    elif self._contact_checker(self._ball_contact_id, self._table_contact_id):  # second check contact with table
                        self._ball_contact_after_hit = True
                        self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                        if self._ball_landing_pos[0] < 0.:  # ball lands on the opponent side
                            self._ball_return_success = True
                        self._terminated = True

            # update ball trajectory & racket trajectory
            self._ball_traj.append(self.data.body("target_ball").xpos.copy())
            self._racket_traj.append(self.data.geom("bat").xpos.copy())

    def ball_racket_contact(self):
        return self._contact_checker(self._ball_contact_id, self._bat_front_id) or \
               self._contact_checker(self._ball_contact_id, self._bat_back_id)

    def step(self, action):
        if not self._id_set:
            self._set_ids()

        unstable_simulation = False
        hit_already = self._hit_ball
        if self._steps == self._goal_switching_step and self.np_random.uniform() < 0.5:
                new_goal_pos = self._generate_goal_pos(random=True)
                new_goal_pos[1] = -new_goal_pos[1]
                self._goal_pos = new_goal_pos
                self.model.body_pos[5] = np.concatenate([self._goal_pos, [0.77]])
                mujoco.mj_forward(self.model, self.data)

        self._update_game_state(action)
        self._steps += 1

        obs = self._get_obs()

        # Compute reward
        if unstable_simulation:
            reward = -25
        else:
            # reward = self._get_reward(self._terminated)
            # hit_now = not hit_already and self._hit_ball
            hit_finish = self._hit_ball and not self.ball_racket_contact()

            if hit_finish:
                # Clean the ball and racket traj before hit
                self._ball_traj = []
                self._racket_traj = []

                # Simulate the rest of the traj
                reward = self._get_reward2(True, False)
                while self._steps < MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER:
                    land_already = self._ball_landing_pos is not None
                    self._update_game_state(np.zeros_like(action))
                    self._steps += 1

                    land_now = (not land_already
                                and self._ball_landing_pos is not None)
                    temp_reward = self._get_reward2(False, land_now)
                    # print(temp_reward)
                    reward += temp_reward

                    # Uncomment the line below to visualize the sim after hit
                    # self.render(mode="human")
            else:
                reward = self._get_reward2(False, False)

        # Update ball landing error
        land_dist_err = np.linalg.norm(self._ball_landing_pos[:-1] - self._goal_pos) \
                            if self._ball_landing_pos is not None else 10.

        info = {
            "hit_ball": self._hit_ball,
            "ball_returned_success": self._ball_return_success,
            "land_dist_error": land_dist_err,
            "is_success": self._ball_return_success and land_dist_err < 0.2,
            "num_steps": self._steps,
        }

        terminated, truncated = self._terminated, self._steps == MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER

        return obs, reward, terminated, truncated, info

class TableTennisWind(TableTennisEnv):
    def __init__(self, ctxt_dim: int = 4, frame_skip: int = 4, **kwargs):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64
        )
        super().__init__(ctxt_dim=ctxt_dim, frame_skip=frame_skip, enable_artificial_wind=True, **kwargs)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:7].copy(),
            self.data.qvel.flat[:7].copy(),
            self.data.joint("tar_x").qpos.copy(),
            self.data.joint("tar_y").qpos.copy(),
            self.data.joint("tar_z").qpos.copy(),
            self.data.joint("tar_x").qvel.copy(),
            self.data.joint("tar_y").qvel.copy(),
            self.data.joint("tar_z").qvel.copy(),
            self._goal_pos.copy(),
        ])
        return obs

class TableTennisGoalSwitching(TableTennisEnv):
    def __init__(self, frame_skip: int = 4, goal_switching_step: int = 99, **kwargs):
        super().__init__(frame_skip=frame_skip, goal_switching_step=goal_switching_step, **kwargs)


class TableTennisRandomInit(TableTennisEnv):
    def __init__(self, ctxt_dim: int = 4, frame_skip: int = 4,
                 random_pos_scale: float = 1.0,
                 random_vel_scale: float = 0.0,
                 **kwargs):
        super().__init__(ctxt_dim=ctxt_dim, frame_skip=frame_skip,
                         random_pos_scale=random_pos_scale,
                         random_vel_scale=random_vel_scale,
                         **kwargs)