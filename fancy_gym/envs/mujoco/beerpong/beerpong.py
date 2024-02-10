import os
from typing import Optional, Any, Dict, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.core import ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco

MAX_EPISODE_STEPS_BEERPONG = 300
FIXED_RELEASE_STEP = 62  # empirically evaluated for frame_skip=2!

# XML Variables
ROBOT_COLLISION_OBJ = ["wrist_palm_link_convex_geom",
                       "wrist_pitch_link_convex_decomposition_p1_geom",
                       "wrist_pitch_link_convex_decomposition_p2_geom",
                       "wrist_pitch_link_convex_decomposition_p3_geom",
                       "wrist_yaw_link_convex_decomposition_p1_geom",
                       "wrist_yaw_link_convex_decomposition_p2_geom",
                       "forearm_link_convex_decomposition_p1_geom",
                       "forearm_link_convex_decomposition_p2_geom",
                       "upper_arm_link_convex_decomposition_p1_geom",
                       "upper_arm_link_convex_decomposition_p2_geom",
                       "shoulder_link_convex_decomposition_p1_geom",
                       "shoulder_link_convex_decomposition_p2_geom",
                       "shoulder_link_convex_decomposition_p3_geom",
                       "base_link_convex_geom", "table_contact_geom"]

CUP_COLLISION_OBJ = ["cup_geom_table3", "cup_geom_table4", "cup_geom_table5", "cup_geom_table6",
                     "cup_geom_table7", "cup_geom_table8", "cup_geom_table9", "cup_geom_table10",
                     "cup_geom_table15", "cup_geom_table16", "cup_geom_table17", "cup_geom1_table8"]


class BeerPongEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self._steps = 0
        # Small Context -> Easier. Todo: Should we do different versions?
        # self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "beerpong_wo_cup.xml")
        # self._cup_pos_min = np.array([-0.32, -2.2])
        # self._cup_pos_max = np.array([0.32, -1.2])

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "beerpong_wo_cup_big_table.xml")
        self._cup_pos_min = np.array([-1.42, -4.05])
        self._cup_pos_max = np.array([1.42, -1.25])

        self._start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
        self._start_vel = np.zeros(7)

        self.release_step = FIXED_RELEASE_STEP

        self.repeat_action = 2
        # TODO: If accessing IDs is easier in the (new) official mujoco bindings, remove this
        self.model = None
        self.geom_id = lambda x: mujoco.mj_name2id(self.model,
                                                   mujoco.mjtObj.mjOBJ_GEOM,
                                                   x)

        # for reward calculation
        self.dists = []
        self.dists_final = []
        self.action_costs = []
        self.ball_ground_contact_first = False
        self.ball_table_contact = False
        self.ball_wall_contact = False
        self.ball_cup_contact = False
        self.ball_in_cup = False
        self.dist_ground_cup = -1  # distance floor to cup if first floor contact

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            self.xml_path,
            frame_skip=1,
            observation_space=self.observation_space,
            **kwargs
        )
        self.render_active = False

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def start_vel(self):
        return self._start_vel

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        self.dists = []
        self.dists_final = []
        self.action_costs = []
        self.ball_ground_contact_first = False
        self.ball_table_contact = False
        self.ball_wall_contact = False
        self.ball_cup_contact = False
        self.ball_in_cup = False
        self.dist_ground_cup = -1  # distance floor to cup if first floor contact
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        # TODO: Ask Max why we need to set the state twice.
        self.set_state(start_pos, init_vel)
        start_pos[7::] = self.data.site("init_ball_pos").xpos.copy()
        self.set_state(start_pos, init_vel)
        xy = self.np_random.uniform(self._cup_pos_min, self._cup_pos_max)
        xyz = np.zeros(3)
        xyz[:2] = xy
        xyz[-1] = 0.840
        self.model.body("cup_table").pos[:] = xyz
        return self._get_obs()

    def step(self, a):
        crash = False
        for _ in range(self.repeat_action):
            applied_action = a + self.data.qfrc_bias[:len(a)].copy() / self.model.actuator_gear[:, 0]
            try:
                self.do_simulation(applied_action, self.frame_skip)
                # self.reward_function.check_contacts(self.sim)   # I assume this is not important?
                if self._steps < self.release_step:
                    self.data.qpos[7::] = self.data.site('init_ball_pos').xpos.copy()
                    self.data.qvel[7::] = self.data.sensor('init_ball_vel').data.copy()
                crash = False
            except Exception as e:
                crash = True

        ob = self._get_obs()

        if not crash:
            reward, reward_infos = self._get_reward(applied_action)
            is_collided = reward_infos['is_collided']  # TODO: Remove if self collision does not make a difference
            terminated = is_collided
            self._steps += 1
        else:
            reward = -30
            terminated = True
            reward_infos = {"success": False, "ball_pos": np.zeros(3), "ball_vel": np.zeros(3), "is_collided": False}

        infos = dict(
            reward=reward,
            action=a,
            q_pos=self.data.qpos[0:7].ravel().copy(),
            q_vel=self.data.qvel[0:7].ravel().copy(), sim_crash=crash,
        )
        infos.update(reward_infos)

        truncated = False

        if self.render_active and self.render_mode=='human':
            self.render()

        return ob, reward, terminated, truncated, infos

    def render(self):
        self.render_active = True
        return super().render()

    def _get_obs(self):
        theta = self.data.qpos.flat[:7].copy()
        theta_dot = self.data.qvel.flat[:7].copy()
        ball_pos = self.data.qpos.flat[7:].copy()
        cup_goal_diff_final = ball_pos - self.data.site("cup_goal_final_table").xpos.copy()
        cup_goal_diff_top = ball_pos - self.data.site("cup_goal_table").xpos.copy()
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            cup_goal_diff_final,
            cup_goal_diff_top,
            self.model.body("cup_table").pos[:2].copy(),
            # [self._steps],  # Use TimeAwareObservation Wrapper instead ....
        ])

    @property
    def dt(self):
        return super(BeerPongEnv, self).dt * self.repeat_action

    def _get_reward(self, action):
        goal_pos = self.data.site("cup_goal_table").xpos
        goal_final_pos = self.data.site("cup_goal_final_table").xpos
        ball_pos = self.data.qpos.flat[7:].copy()
        ball_vel = self.data.qvel.flat[7:].copy()

        self._check_contacts()
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.dist_ground_cup = np.linalg.norm(ball_pos - goal_pos) \
            if self.ball_ground_contact_first and self.dist_ground_cup == -1 else self.dist_ground_cup
        action_cost = np.sum(np.square(action))
        self.action_costs.append(np.copy(action_cost))
        # # ##################### Reward function which does not force to bounce once on the table (quad dist) #########

        # Is this needed?
        # self._is_collided = self._check_collision_with_itself([self.geom_id(name) for name in CUP_COLLISION_OBJ])

        if self._steps == MAX_EPISODE_STEPS_BEERPONG - 1:  # or self._is_collided:
            min_dist = np.min(self.dists)
            final_dist = self.dists_final[-1]
            if self.ball_ground_contact_first:
                min_dist_coeff, final_dist_coeff, ground_contact_dist_coeff, rew_offset = 1, 0.5, 2, -4
            else:
                if not self.ball_in_cup:
                    if not self.ball_table_contact and not self.ball_cup_contact and not self.ball_wall_contact:
                        min_dist_coeff, final_dist_coeff, ground_contact_dist_coeff, rew_offset = 1, 0.5, 0, -4
                    else:
                        min_dist_coeff, final_dist_coeff, ground_contact_dist_coeff, rew_offset = 1, 0.5, 0, -2
                else:
                    min_dist_coeff, final_dist_coeff, ground_contact_dist_coeff, rew_offset = 0, 1, 0, 0
            action_cost = 1e-4 * np.mean(action_cost)
            reward = rew_offset - min_dist_coeff * min_dist ** 2 - final_dist_coeff * final_dist ** 2 - \
                action_cost - ground_contact_dist_coeff * self.dist_ground_cup ** 2
            # release step punishment
            min_time_bound = 0.1
            max_time_bound = 1.0
            release_time = self.release_step * self.dt
            release_time_rew = int(release_time < min_time_bound) * (-30 - 10 * (release_time - min_time_bound) ** 2) + \
                int(release_time > max_time_bound) * (-30 - 10 * (release_time - max_time_bound) ** 2)
            reward += release_time_rew
            success = self.ball_in_cup
        else:
            action_cost = 1e-2 * action_cost
            reward = - action_cost
            success = False
        # ##############################################################################################################
        infos = {"success": success, "ball_pos": ball_pos.copy(),
                 "ball_vel": ball_vel.copy(), "action_cost": action_cost, "task_reward": reward,
                 "table_contact_first": int(not self.ball_ground_contact_first),
                 "is_collided": False}  # TODO: Check if is collided is needed
        return reward, infos

    def _check_contacts(self):
        # TODO proper contact detection using sensors?
        if not self.ball_table_contact:
            self.ball_table_contact = self._check_collision({self.geom_id("ball_geom")},
                                                            {self.geom_id("table_contact_geom")})
        if not self.ball_cup_contact:
            self.ball_cup_contact = self._check_collision({self.geom_id("ball_geom")},
                                                          {self.geom_id(name) for name in CUP_COLLISION_OBJ})
        if not self.ball_wall_contact:
            self.ball_wall_contact = self._check_collision({self.geom_id("ball_geom")},
                                                           {self.geom_id("wall")})
        if not self.ball_in_cup:
            self.ball_in_cup = self._check_collision({self.geom_id("ball_geom")},
                                                     {self.geom_id("cup_base_table_contact")})
        if not self.ball_ground_contact_first:
            if not self.ball_table_contact and not self.ball_cup_contact and not self.ball_wall_contact \
                    and not self.ball_in_cup:
                self.ball_ground_contact_first = self._check_collision({self.geom_id("ball_geom")},
                                                                       {self.geom_id("ground")})

    def _check_collision(self, id_set_1, id_set_2=None):
        """
        Checks if id_set1 has a collision with id_set2.
        If id_set_2 is set to None, it will check for a collision with itself (id_set_1).
        """
        collision_id_set = id_set_2 - id_set_1 if id_set_2 is not None else id_set_1
        for coni in range(self.data.ncon):
            con = self.data.contact[coni]
            if ((con.geom1 in id_set_1 and con.geom2 in collision_id_set) or
                    (con.geom2 in id_set_1 and con.geom1 in collision_id_set)):
                return True
        return False


class BeerPongEnvStepBasedEpisodicReward(BeerPongEnv):

    def step(self, a):
        if self._steps < FIXED_RELEASE_STEP:
            return super(BeerPongEnvStepBasedEpisodicReward, self).step(a)
        else:
            reward = 0
            terminated, truncated = True, False
            while self._steps < MAX_EPISODE_STEPS_BEERPONG:
                obs, sub_reward, terminated, truncated, infos = super(BeerPongEnvStepBasedEpisodicReward, self).step(
                    np.zeros(a.shape))
                reward += sub_reward
        return obs, reward, terminated, truncated, infos
