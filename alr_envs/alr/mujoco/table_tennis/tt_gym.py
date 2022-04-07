import os

import numpy as np
import mujoco_py
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

from alr_envs.alr.mujoco.table_tennis.tt_utils import ball_init
from alr_envs.alr.mujoco.table_tennis.tt_reward import TT_Reward

#TODO: Check for simulation stability. Make sure the code runs even for sim crash

MAX_EPISODE_STEPS = 1375      # (1.25 + 1.5) /0.002
BALL_NAME_CONTACT = "target_ball_contact"
BALL_NAME = "target_ball"
TABLE_NAME = 'table_tennis_table'
FLOOR_NAME = 'floor'
PADDLE_CONTACT_1_NAME = 'bat'
PADDLE_CONTACT_2_NAME = 'bat_back'
RACKET_NAME = 'bat'
# CONTEXT_RANGE_BOUNDS_2DIM = np.array([[-1.2, -0.6], [-0.2, 0.6]])
CONTEXT_RANGE_BOUNDS_2DIM = np.array([[-1.2, -0.6], [-0.2, 0.0]])
CONTEXT_RANGE_BOUNDS_4DIM = np.array([[-1.35, -0.75, -1.25, -0.75], [-0.1, 0.75, -0.1, 0.75]])


class TTEnvGym(MujocoEnv, utils.EzPickle):

    def __init__(self, ctxt_dim=2, fixed_goal=False):
        model_path = os.path.join(os.path.dirname(__file__), "xml", 'table_tennis_env.xml')

        self.ctxt_dim = ctxt_dim
        self.fixed_goal = fixed_goal
        if ctxt_dim == 2:
            self.context_range_bounds = CONTEXT_RANGE_BOUNDS_2DIM
            if self.fixed_goal:
                self.goal = np.array([-1, -0.1, 0])
            else:
                self.goal = np.zeros(3)  # 2 x,y + 1z
        elif ctxt_dim == 4:
            self.context_range_bounds = CONTEXT_RANGE_BOUNDS_4DIM
            self.goal = np.zeros(3)
        else:
            raise ValueError("either 2 or 4 dimensional Contexts available")

        # has no effect as it is overwritten in init of super
        # action_space_low = np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2])
        # action_space_high = np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2])
        # self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype='float64')

        self.time_steps = 0
        self.init_qpos_tt = np.array([0, 0, 0, 1.5, 0, 0, 1.5, 0, 0, 0])
        self.init_qvel_tt = np.zeros(10)

        self.reward_func = TT_Reward(self.ctxt_dim)
        self.ball_landing_pos = None
        self.hit_ball = False
        self.ball_contact_after_hit = False
        self._ids_set = False
        super(TTEnvGym, self).__init__(model_path=model_path, frame_skip=1)
        self.ball_id = self.sim.model._body_name2id[BALL_NAME]  # find the proper -> not protected func.
        self.ball_contact_id = self.sim.model._geom_name2id[BALL_NAME_CONTACT]
        self.table_contact_id = self.sim.model._geom_name2id[TABLE_NAME]
        self.floor_contact_id = self.sim.model._geom_name2id[FLOOR_NAME]
        self.paddle_contact_id_1 = self.sim.model._geom_name2id[PADDLE_CONTACT_1_NAME]  # check if we need both or only this
        self.paddle_contact_id_2 = self.sim.model._geom_name2id[PADDLE_CONTACT_2_NAME]  # check if we need both or only this
        self.racket_id = self.sim.model._geom_name2id[RACKET_NAME]


    def _set_ids(self):
        self.ball_id = self.sim.model._body_name2id[BALL_NAME]  # find the proper -> not protected func.
        self.table_contact_id = self.sim.model._geom_name2id[TABLE_NAME]
        self.floor_contact_id = self.sim.model._geom_name2id[FLOOR_NAME]
        self.paddle_contact_id_1 = self.sim.model._geom_name2id[PADDLE_CONTACT_1_NAME]  # check if we need both or only this
        self.paddle_contact_id_2 = self.sim.model._geom_name2id[PADDLE_CONTACT_2_NAME]  # check if we need both or only this
        self.racket_id = self.sim.model._geom_name2id[RACKET_NAME]
        self.ball_contact_id = self.sim.model._geom_name2id[BALL_NAME_CONTACT]
        self._ids_set = True

    def _get_obs(self):
        ball_pos = self.sim.data.body_xpos[self.ball_id][:2].copy()
        goal_pos = self.goal[:2].copy()
        obs = np.concatenate([self.sim.data.qpos[:7].copy(),  # 7 joint positions
                              ball_pos,
                              goal_pos])
        return obs

    def sample_context(self):
        return self.np_random.uniform(self.context_range_bounds[0], self.context_range_bounds[1], size=self.ctxt_dim)

    def reset_model(self):
        self.set_state(self.init_qpos_tt, self.init_qvel_tt)    # reset to initial sim state
        self.time_steps = 0
        self.ball_landing_pos = None
        self.hit_ball = False
        self.ball_contact_after_hit = False
        if self.fixed_goal:
            self.goal = self.goal[:2]
        else:
            self.goal = self.sample_context()[:2]
        if self.ctxt_dim == 2:
            initial_ball_state = ball_init(random=False)  # fixed velocity, fixed position
        elif self.ctxt_dim == 4:
            initial_ball_state = ball_init(random=False)#raise NotImplementedError

        self.sim.data.set_joint_qpos('tar:x', initial_ball_state[0])
        self.sim.data.set_joint_qpos('tar:y', initial_ball_state[1])
        self.sim.data.set_joint_qpos('tar:z', initial_ball_state[2])

        self.sim.data.set_joint_qvel('tar:x', initial_ball_state[3])
        self.sim.data.set_joint_qvel('tar:y', initial_ball_state[4])
        self.sim.data.set_joint_qvel('tar:z', initial_ball_state[5])

        z_extended_goal_pos = np.concatenate((self.goal[:2], [0.77]))
        self.goal = z_extended_goal_pos
        self.sim.model.body_pos[5] = self.goal[:3]          # Desired Landing Position, Yellow
        self.sim.model.body_pos[3] = np.array([0, 0, 0.5])  # Outgoing Ball Landing Position, Green
        self.sim.model.body_pos[4] = np.array([0, 0, 0.5])  # Incoming Ball Landing Position, Red
        self.sim.forward()

        self.reward_func.reset(self.goal)                   # reset the reward function
        self.n_step = 0
        return self._get_obs()

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            collision = con.geom1 == id_1 and con.geom2 == id_2
            collision_trans = con.geom1 == id_2 and con.geom2 == id_1
            if collision or collision_trans:
                return True
        return False

    def step(self, action):
        if not self._ids_set:
            self._set_ids()
        done = False
        episode_end = False if self.time_steps + 1 < MAX_EPISODE_STEPS else True
        if not self.hit_ball:
            self.hit_ball = self._contact_checker(self.ball_contact_id, self.paddle_contact_id_1) # check for one side
            if not self.hit_ball:
                self.hit_ball = self._contact_checker(self.ball_contact_id, self.paddle_contact_id_2) # check for other side
        if self.hit_ball:
            if not self.ball_contact_after_hit:
                if self._contact_checker(self.ball_contact_id, self.floor_contact_id):  # first check contact with floor
                    self.ball_contact_after_hit = True
                    self.ball_landing_pos = self.sim.data.body_xpos[self.ball_id]
                elif self._contact_checker(self.ball_contact_id, self.table_contact_id): # second check contact with table
                    self.ball_contact_after_hit = True
                    self.ball_landing_pos = self.sim.data.body_xpos[self.ball_id]
        c_ball_pos = self.sim.data.body_xpos[self.ball_id]
        racket_pos = self.sim.data.geom_xpos[self.racket_id]        # TODO: use this to reach out the position of the paddle?
        if self.ball_landing_pos is not None:
            done = True
            episode_end =True
        reward = self.reward_func.get_reward(episode_end, c_ball_pos, racket_pos, self.hit_ball, self.ball_landing_pos)
        self.time_steps += 1
        # gravity compensation on joints:
        #action += self.sim.data.qfrc_bias[:7].copy()
        try:
            self.do_simulation(action, self.frame_skip)
        except mujoco_py.MujocoException as e:
            print('Simulation got unstable returning')
            done = True
            reward = -25
        ob = self._get_obs()
        info = {"hit_ball": self.hit_ball,
                "q_pos": np.copy(self.sim.data.qpos[:7]),
                "ball_pos": np.copy(self.sim.data.qpos[7:])}
        self.n_step += 1
        return ob, reward, done, info # might add some information here ....

    def set_context(self, context):
        old_state = self.sim.get_state()
        qpos = old_state.qpos.copy()
        qvel = old_state.qvel.copy()
        self.set_state(qpos, qvel)
        self.goal = context
        z_extended_goal_pos = np.concatenate((self.goal[:self.ctxt_dim], [0.77]))
        if self.ctxt_dim == 4:
            z_extended_goal_pos = np.concatenate((z_extended_goal_pos, [0.77]))
        self.goal = z_extended_goal_pos
        self.sim.model.body_pos[5] = self.goal[:3]      # TODO: Missing: Setting the desired incomoing landing position
        self.sim.forward()
        return self._get_obs()
