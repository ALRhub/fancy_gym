import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import check_init_state_validity, magnus_force

import mujoco

MAX_EPISODE_STEPS_TABLE_TENNIS = 250

CONTEXT_BOUNDS_2DIMS = np.array([[-1.0, -0.65], [-0.2, 0.65]])
CONTEXT_BOUNDS_4DIMS = np.array([[-1.0, -0.65, -1.0, -0.65],
                                 [-0.2, 0.65, -0.2, 0.65]])
CONTEXT_BOUNDS_SWICHING = np.array([[-1.0, -0.65, -1.0, 0.],
                                    [-0.2, 0.65, -0.2, 0.65]])


class TableTennisEnv(MujocoEnv, utils.EzPickle):
    """
    7 DoF table tennis environment
    """

    def __init__(self, ctxt_dim: int = 4, frame_skip: int = 4,
                 enable_switching_goal: bool = False,
                 enable_wind: bool = False, enable_magnus: bool = False,
                 enable_air: bool = False):
        utils.EzPickle.__init__(**locals())
        self._steps = 0

        self._hit_ball = False
        self._ball_land_on_table = False
        self._ball_contact_after_hit = False
        self._ball_return_success = False
        self._ball_landing_pos = None
        self._init_ball_state = None
        self._episode_end = False

        self._id_set = False

        # reward calculation
        self.ball_landing_pos = None
        self._goal_pos = np.zeros(2)
        self._ball_traj = []
        self._racket_traj = []


        self._enable_goal_switching = enable_switching_goal

        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "xml", "table_tennis_env.xml"),
                           frame_skip=frame_skip,
                           mujoco_bindings="mujoco")
        if ctxt_dim == 2:
            self.context_bounds = CONTEXT_BOUNDS_2DIMS
        elif ctxt_dim == 4:
            self.context_bounds = CONTEXT_BOUNDS_4DIMS
            if self._enable_goal_switching:
                self.context_bounds = CONTEXT_BOUNDS_SWICHING
        else:
            raise NotImplementedError

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # complex dynamics settings
        if enable_air:
            self.model.opt.density = 1.225
            self.model.opt.viscosity = 2.27e-5

        self._enable_wind = enable_wind
        self._enable_magnus = enable_magnus
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

        if self._enable_goal_switching:
            if self._steps == 45 and self.np_random.uniform(0, 1) < 0.5:
                self._goal_pos[1] = -self._goal_pos[1]
                self.model.body_pos[5] = np.concatenate([self._goal_pos, [0.77]])
                mujoco.mj_forward(self.model, self.data)

        for _ in range(self.frame_skip):
            try:
                self.do_simulation(action, 1)
            except Exception as e:
                print("Simulation get unstable return with MujocoException: ", e)
                unstable_simulation = True
                self._episode_end = True
                break

            if not self._hit_ball:
                self._hit_ball = self._contact_checker(self._ball_contact_id, self._bat_front_id) or \
                                self._contact_checker(self._ball_contact_id, self._bat_back_id)
                if not self._hit_ball:
                    ball_land_on_floor_no_hit = self._contact_checker(self._ball_contact_id, self._floor_contact_id)
                    if ball_land_on_floor_no_hit:
                        self._ball_landing_pos = self.data.body("target_ball").xpos.copy()
                        self._episode_end = True
            if self._hit_ball and not self._ball_contact_after_hit:
                if not self._ball_contact_after_hit:
                    if self._contact_checker(self._ball_contact_id, self._floor_contact_id):  # first check contact with floor
                        self._ball_contact_after_hit = True
                        self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                        self._episode_end = True
                    elif self._contact_checker(self._ball_contact_id, self._table_contact_id):  # second check contact with table
                        self._ball_contact_after_hit = True
                        self._ball_landing_pos = self.data.geom("target_ball_contact").xpos.copy()
                        if self._ball_landing_pos[0] < 0.:  # ball lands on the opponent side
                            self._ball_return_success = True
                        self._episode_end = True

            # update ball trajectory & racket trajectory
            self._ball_traj.append(self.data.body("target_ball").xpos.copy())
            self._racket_traj.append(self.data.geom("bat").xpos.copy())

        self._steps += 1
        self._episode_end = True if self._steps >= MAX_EPISODE_STEPS_TABLE_TENNIS else self._episode_end

        reward = -25 if unstable_simulation else self._get_reward(self._episode_end)

        land_dist_err = np.linalg.norm(self._ball_landing_pos[:-1] - self._goal_pos) \
                            if self._ball_landing_pos is not None else 10.

        return self._get_obs(), reward, self._episode_end, {
            "hit_ball": self._hit_ball,
            "ball_returned_success": self._ball_return_success,
            "land_dist_error": land_dist_err,
            "is_success": self._ball_return_success and land_dist_err < 0.2,
            "num_steps": self._steps,
        }

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]
            if (con.geom1 == id_1 and con.geom2 == id_2) or (con.geom1 == id_2 and con.geom2 == id_1):
                return True
        return False

    def reset_model(self):
        self._steps = 0
        self._init_ball_state = self._generate_valid_init_ball(random_pos=False, random_vel=False)
        self._goal_pos = self._generate_goal_pos(random=False)
        self.data.joint("tar_x").qpos = self._init_ball_state[0]
        self.data.joint("tar_y").qpos = self._init_ball_state[1]
        self.data.joint("tar_z").qpos = self._init_ball_state[2]
        self.data.joint("tar_x").qvel = self._init_ball_state[3]
        self.data.joint("tar_y").qvel = self._init_ball_state[4]
        self.data.joint("tar_z").qvel = self._init_ball_state[5]

        self.model.body_pos[5] = np.concatenate([self._goal_pos, [0.77]])

        self.data.qpos[:7] = np.array([0., 0., 0., 1.5, 0., 0., 1.5])

        mujoco.mj_forward(self.model, self.data)

        if self._enable_wind:
            self._wind_vel[1] = self.np_random.uniform(low=-5, high=5, size=1)
            self.model.opt.wind[:3] = self._wind_vel

        self._hit_ball = False
        self._ball_land_on_table = False
        self._ball_contact_after_hit = False
        self._ball_return_success = False
        self._ball_landing_pos = None
        self._episode_end = False
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
            # self.data.joint("tar_x").qvel.copy(),
            # self.data.joint("tar_y").qvel.copy(),
            # self.data.joint("tar_z").qvel.copy(),
            # self.data.body("target_ball").xvel.copy(),
            self._goal_pos.copy(),
        ])
        return obs

    def get_obs(self):
        return self._get_obs()

    def _get_reward(self, episode_end):
        if not episode_end:
            return 0
        else:
            min_r_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj) - np.array(self._racket_traj), axis=1))
            if not self._hit_ball:
                return 0.2 * (1 - np.tanh(min_r_b_dist**2))
            else:
                if self._ball_landing_pos is None:
                    min_b_des_b_dist = np.min(np.linalg.norm(np.array(self._ball_traj)[:,:2] - self._goal_pos[:2], axis=1))
                    return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + (1 - np.tanh(min_b_des_b_dist**2))
                else:
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
        while not check_init_state_validity(init_ball_state):
            init_ball_state = self._generate_random_ball(random_pos=random_pos, random_vel=random_vel)
        return init_ball_state

def plot_ball_traj(x_traj, y_traj, z_traj):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj, y_traj, z_traj)
    plt.show()

def plot_ball_traj_2d(x_traj, y_traj):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_traj, y_traj)
    plt.show()

def plot_single_axis(traj, title):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(traj)
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    env = TableTennisEnv(enable_air=True)
    # env_with_air = TableTennisEnv(enable_air=True)
    for _ in range(1):
        obs1 = env.reset()
        # obs2 = env_with_air.reset()
        x_pos = []
        y_pos = []
        z_pos = []
        x_vel = []
        y_vel = []
        z_vel = []
        for _ in range(2000):
            obs, reward, done, info = env.step(np.zeros(7))
            # _, _, _, _ = env_no_air.step(np.zeros(7))
            x_pos.append(env.data.joint("tar_x").qpos[0])
            y_pos.append(env.data.joint("tar_y").qpos[0])
            z_pos.append(env.data.joint("tar_z").qpos[0])
            x_vel.append(env.data.joint("tar_x").qvel[0])
            y_vel.append(env.data.joint("tar_y").qvel[0])
            z_vel.append(env.data.joint("tar_z").qvel[0])
            # print(reward)
            if done:
                # plot_ball_traj_2d(x_pos, y_pos)
                plot_single_axis(x_pos, title="x_vel without air")
                break
