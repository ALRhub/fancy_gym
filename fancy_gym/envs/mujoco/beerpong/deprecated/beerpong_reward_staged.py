import numpy as np


class BeerPongReward:
    def __init__(self):

        self.robot_collision_objects = ["wrist_palm_link_convex_geom",
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

        self.cup_collision_objects = ["cup_geom_table3", "cup_geom_table4", "cup_geom_table5", "cup_geom_table6",
                                      "cup_geom_table7", "cup_geom_table8", "cup_geom_table9", "cup_geom_table10",
                                      "cup_geom_table15",
                                      "cup_geom_table16",
                                      "cup_geom_table17", "cup_geom1_table8",
                                      ]

        self.dists = None
        self.dists_final = None
        self.action_costs = None
        self.ball_ground_contact_first = False
        self.ball_table_contact = False
        self.ball_wall_contact = False
        self.ball_cup_contact = False
        self.ball_in_cup = False
        self.dist_ground_cup = -1  # distance floor to cup if first floor contact

        # IDs
        self.ball_collision_id = None
        self.table_collision_id = None
        self.wall_collision_id = None
        self.cup_table_collision_id = None
        self.ground_collision_id = None
        self.cup_collision_ids = None
        self.robot_collision_ids = None
        self.reset()
        self.is_initialized = False

    def reset(self):
        self.dists = []
        self.dists_final = []
        self.action_costs = []
        self.ball_ground_contact_first = False
        self.ball_table_contact = False
        self.ball_wall_contact = False
        self.ball_cup_contact = False
        self.ball_in_cup = False
        self.dist_ground_cup = -1  # distance floor to cup if first floor contact

    def initialize(self, env):

        if not self.is_initialized:
            self.is_initialized = True
            # TODO: Find a more elegant way to acces to the geom ids in each step -> less code
            self.ball_collision_id = {env.model.geom_name2id("ball_geom")}
            # self.ball_collision_id = env.model.geom_name2id("ball_geom")
            self.table_collision_id = {env.model.geom_name2id("table_contact_geom")}
            # self.table_collision_id = env.model.geom_name2id("table_contact_geom")
            self.wall_collision_id = {env.model.geom_name2id("wall")}
            # self.wall_collision_id = env.model.geom_name2id("wall")
            self.cup_table_collision_id = {env.model.geom_name2id("cup_base_table_contact")}
            # self.cup_table_collision_id = env.model.geom_name2id("cup_base_table_contact")
            self.ground_collision_id = {env.model.geom_name2id("ground")}
            # self.ground_collision_id = env.model.geom_name2id("ground")
            self.cup_collision_ids = {env.model.geom_name2id(name) for name in self.cup_collision_objects}
            # self.cup_collision_ids = [env.model.geom_name2id(name) for name in self.cup_collision_objects]
            self.robot_collision_ids = [env.model.geom_name2id(name) for name in self.robot_collision_objects]

    def compute_reward(self, env, action):

        goal_pos = env.data.get_site_xpos("cup_goal_table")
        ball_pos = env.data.get_body_xpos("ball")
        ball_vel = env.data.get_body_xvelp("ball")
        goal_final_pos = env.data.get_site_xpos("cup_goal_final_table")

        self.check_contacts(env.sim)
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.dist_ground_cup = np.linalg.norm(ball_pos - goal_pos) \
            if self.ball_ground_contact_first and self.dist_ground_cup == -1 else self.dist_ground_cup
        action_cost = np.sum(np.square(action))
        self.action_costs.append(np.copy(action_cost))
        # # ##################### Reward function which does not force to bounce once on the table (quad dist) #########

        # Is this needed?
        # self._is_collided = self._check_collision_with_itself(env.sim, self.robot_collision_ids)

        if env._steps == env.ep_length - 1:  # or self._is_collided:
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
            release_time = env.release_step * env.dt
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

    def check_contacts(self, sim):
        if not self.ball_table_contact:
            self.ball_table_contact = self._check_collision(sim, self.ball_collision_id, self.table_collision_id)
        if not self.ball_cup_contact:
            self.ball_cup_contact = self._check_collision(sim, self.ball_collision_id, self.cup_collision_ids)
        if not self.ball_wall_contact:
            self.ball_wall_contact = self._check_collision(sim, self.ball_collision_id, self.wall_collision_id)
        if not self.ball_in_cup:
            self.ball_in_cup = self._check_collision(sim, self.ball_collision_id, self.cup_table_collision_id)
        if not self.ball_ground_contact_first:
            if not self.ball_table_contact and not self.ball_cup_contact and not self.ball_wall_contact \
                    and not self.ball_in_cup:
                self.ball_ground_contact_first = self._check_collision(sim, self.ball_collision_id,
                                                                       self.ground_collision_id)

    # Checks if id_set1 has a collision with id_set2
    def _check_collision(self, sim, id_set_1, id_set_2):
        """
        If id_set_2 is set to None, it will check for a collision with itself (id_set_1).
        """
        collision_id_set = id_set_2 - id_set_1 if id_set_2 is not None else id_set_1
        for coni in range(sim.data.ncon):
            con = sim.data.contact[coni]
            if ((con.geom1 in id_set_1 and con.geom2 in collision_id_set) or
                    (con.geom2 in id_set_1 and con.geom1 in collision_id_set)):
                return True
        return False

    # def _check_collision_with_itself(self, sim, collision_ids):
    #     col_1, col_2 = False, False
    #     for j, id in enumerate(collision_ids):
    #         col_1 = self._check_collision_with_set_of_objects(sim, id, collision_ids[:j])
    #         if j != len(collision_ids) - 1:
    #             col_2 = self._check_collision_with_set_of_objects(sim, id, collision_ids[j + 1:])
    #         else:
    #             col_2 = False
    #     collision = True if col_1 or col_2 else False
    #     return collision

    ### This function will not be needed if we really do not need to check for collision with itself
    # def _check_collision_with_set_of_objects(self, sim, id_1, id_list):
    #     for coni in range(0, sim.data.ncon):
    #         con = sim.data.contact[coni]
    #
    #         collision = con.geom1 in id_list and con.geom2 == id_1
    #         collision_trans = con.geom1 == id_1 and con.geom2 in id_list
    #
    #         if collision or collision_trans:
    #             return True
    #     return False
