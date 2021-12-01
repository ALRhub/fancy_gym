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
                                      # "cup_base_table", "cup_base_table_contact",
                                      "cup_geom_table15",
                                      "cup_geom_table16",
                                      "cup_geom_table17", "cup_geom1_table8",
                                      # "cup_base_table_contact",
                                      # "cup_base_table"
                                      ]


        self.ball_traj = None
        self.dists = None
        self.dists_final = None
        self.costs = None
        self.action_costs = None
        self.angle_rewards = None
        self.cup_angles = None
        self.cup_z_axes = None
        self.collision_penalty = 500
        self.reset(None)

    def reset(self, noisy):
        self.ball_traj = []
        self.dists = []
        self.dists_final = []
        self.costs = []
        self.action_costs = []
        self.angle_rewards = []
        self.cup_angles = []
        self.cup_z_axes = []
        self.ball_ground_contact = False
        self.ball_table_contact = False
        self.ball_wall_contact = False
        self.ball_cup_contact = False
        self.noisy_bp = noisy
        self._t_min_final_dist = -1

    def compute_reward(self, env, action):
        self.ball_id = env.sim.model._body_name2id["ball"]
        self.ball_collision_id = env.sim.model._geom_name2id["ball_geom"]
        self.goal_id = env.sim.model._site_name2id["cup_goal_table"]
        self.goal_final_id = env.sim.model._site_name2id["cup_goal_final_table"]
        self.cup_collision_ids = [env.sim.model._geom_name2id[name] for name in self.cup_collision_objects]
        self.cup_table_id = env.sim.model._body_name2id["cup_table"]
        self.table_collision_id = env.sim.model._geom_name2id["table_contact_geom"]
        self.wall_collision_id = env.sim.model._geom_name2id["wall"]
        self.cup_table_collision_id = env.sim.model._geom_name2id["cup_base_table_contact"]
        self.init_ball_pos_site_id = env.sim.model._site_name2id["init_ball_pos_site"]
        self.ground_collision_id = env.sim.model._geom_name2id["ground"]
        self.robot_collision_ids = [env.sim.model._geom_name2id[name] for name in self.robot_collision_objects]

        goal_pos = env.sim.data.site_xpos[self.goal_id]
        ball_pos = env.sim.data.body_xpos[self.ball_id]
        ball_vel = env.sim.data.body_xvelp[self.ball_id]
        goal_final_pos = env.sim.data.site_xpos[self.goal_final_id]
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))

        action_cost = np.sum(np.square(action))
        self.action_costs.append(action_cost)

        self._is_collided = self._check_collision_with_itself(env.sim, self.robot_collision_ids)
        if env._steps == env.ep_length - 1 or self._is_collided:

            min_dist = np.min(self.dists)
            ball_table_bounce = self._check_collision_single_objects(env.sim, self.ball_collision_id,
                                                                     self.table_collision_id)
            ball_cup_table_cont = self._check_collision_with_set_of_objects(env.sim, self.ball_collision_id,
                                                                            self.cup_collision_ids)
            ball_wall_cont = self._check_collision_single_objects(env.sim, self.ball_collision_id,
                                                                  self.wall_collision_id)
            ball_in_cup = self._check_collision_single_objects(env.sim, self.ball_collision_id,
                                                               self.cup_table_collision_id)
            if not ball_in_cup:
                cost_offset = 2
                if not ball_cup_table_cont and not ball_table_bounce and not ball_wall_cont:
                    cost_offset += 2
                cost = cost_offset + min_dist ** 2 + 0.5 * self.dists_final[-1] ** 2 + 1e-7 * action_cost
            else:
                cost = self.dists_final[-1] ** 2 + 1.5 * action_cost * 1e-7

            reward = - 1 * cost - self.collision_penalty * int(self._is_collided)
            success = ball_in_cup
            crash = self._is_collided
        else:
            reward = - 1e-7 * action_cost
            cost = 0
            success = False
            crash = False

        infos = {}
        infos["success"] = success
        infos["is_collided"] = self._is_collided
        infos["ball_pos"] = ball_pos.copy()
        infos["ball_vel"] = ball_vel.copy()
        infos["action_cost"] = 5e-4 * action_cost
        infos["task_cost"] = cost

        return reward, infos

    def get_cost_offset(self):
        if self.ball_ground_contact:
            return 200

        if not self.ball_table_contact:
            return 100

        if not self.ball_in_cup:
            return 50

        if self.ball_in_cup and self.ball_cup_contact and not self.noisy_bp:
            return 10

        return 0

    def _check_collision_single_objects(self, sim, id_1, id_2):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 == id_1 and con.geom2 == id_2
            collision_trans = con.geom1 == id_2 and con.geom2 == id_1

            if collision or collision_trans:
                return True
        return False

    def _check_collision_with_itself(self, sim, collision_ids):
        col_1, col_2 = False, False
        for j, id in enumerate(collision_ids):
            col_1 = self._check_collision_with_set_of_objects(sim, id, collision_ids[:j])
            if j != len(collision_ids) - 1:
                col_2 = self._check_collision_with_set_of_objects(sim, id, collision_ids[j + 1:])
            else:
                col_2 = False
        collision = True if col_1 or col_2 else False
        return collision

    def _check_collision_with_set_of_objects(self, sim, id_1, id_list):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 in id_list and con.geom2 == id_1
            collision_trans = con.geom1 == id_1 and con.geom2 in id_list

            if collision or collision_trans:
                return True
        return False