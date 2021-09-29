import numpy as np
from alr_envs.alr.mujoco import alr_reward_fct


class BeerpongReward(alr_reward_fct.AlrReward):
    def __init__(self, sim, sim_time):

        self.sim = sim
        self.sim_time = sim_time

        self.collision_objects = ["cup_geom1", "cup_geom2", "wrist_palm_link_convex_geom",
                                  "wrist_pitch_link_convex_decomposition_p1_geom",
                                  "wrist_pitch_link_convex_decomposition_p2_geom",
                                  "wrist_pitch_link_convex_decomposition_p3_geom",
                                  "wrist_yaw_link_convex_decomposition_p1_geom",
                                  "wrist_yaw_link_convex_decomposition_p2_geom",
                                  "forearm_link_convex_decomposition_p1_geom",
                                  "forearm_link_convex_decomposition_p2_geom"]

        self.ball_id = None
        self.ball_collision_id = None
        self.goal_id = None
        self.goal_final_id = None
        self.collision_ids = None

        self.ball_traj = None
        self.dists = None
        self.dists_ctxt = None
        self.dists_final = None
        self.costs = None

        self.reset(None)

    def reset(self, context):
        self.ball_traj = np.zeros(shape=(self.sim_time, 3))
        self.dists = []
        self.dists_ctxt = []
        self.dists_final = []
        self.costs = []
        self.action_costs = []
        self.context = context
        self.ball_in_cup = False
        self.dist_ctxt = 5
        self.bounce_dist = 2
        self.min_dist = 2
        self.dist_final = 2
        self.table_contact = False

        self.ball_id = self.sim.model._body_name2id["ball"]
        self.ball_collision_id = self.sim.model._geom_name2id["ball_geom"]
        self.cup_robot_id = self.sim.model._site_name2id["cup_robot_final"]
        self.goal_id = self.sim.model._site_name2id["cup_goal_table"]
        self.goal_final_id = self.sim.model._site_name2id["cup_goal_final_table"]
        self.collision_ids = [self.sim.model._geom_name2id[name] for name in self.collision_objects]
        self.cup_table_id = self.sim.model._body_name2id["cup_table"]
        self.bounce_table_id = self.sim.model._site_name2id["bounce_table"]

    def compute_reward(self, action, sim, step):
        action_cost = np.sum(np.square(action))
        self.action_costs.append(action_cost)

        stop_sim = False
        success = False

        if self.check_collision(sim):
            reward = - 1e-2 * action_cost - 10
            stop_sim = True
            return reward, success, stop_sim

        # Compute the current distance from the ball to the inner part of the cup
        goal_pos = sim.data.site_xpos[self.goal_id]
        ball_pos = sim.data.body_xpos[self.ball_id]
        bounce_pos = sim.data.site_xpos[self.bounce_table_id]
        goal_final_pos = sim.data.site_xpos[self.goal_final_id]
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.ball_traj[step, :] = ball_pos

        ball_in_cup = self.check_ball_in_cup(sim, self.ball_collision_id)
        table_contact = self.check_ball_table_contact(sim, self.ball_collision_id)

        if table_contact and not self.table_contact:
            self.bounce_dist = np.minimum((np.linalg.norm(bounce_pos - ball_pos)), 2)
            self.table_contact = True

        if step == self.sim_time - 1:
            min_dist = np.min(self.dists)
            self.min_dist = min_dist
            dist_final = self.dists_final[-1]
            self.dist_final = dist_final

            cost = 0.33 * min_dist + 0.33 * dist_final + 0.33 * self.bounce_dist
            reward = np.exp(-2 * cost) - 1e-2 * action_cost
            success = self.bounce_dist < 0.05 and dist_final < 0.05 and ball_in_cup
        else:
            reward = - 1e-2 * action_cost
            success = False

        return reward, success, stop_sim

    def _get_stage_wise_cost(self, ball_in_cup, min_dist, dist_final, dist_to_ctxt):
        if not ball_in_cup:
            cost = 3 + 2*(0.5 * min_dist**2 + 0.5 * dist_final**2)
        else:
            cost = 2 * dist_to_ctxt ** 2
            print('Context Distance:', dist_to_ctxt)
        return cost

    def check_ball_table_contact(self, sim, ball_collision_id):
        table_collision_id = sim.model._geom_name2id["table_contact_geom"]
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]
            collision = con.geom1 == table_collision_id and con.geom2 == ball_collision_id
            collision_trans = con.geom1 == ball_collision_id and con.geom2 == table_collision_id

            if collision or collision_trans:
                return True
        return False

    def check_ball_in_cup(self, sim, ball_collision_id):
        cup_base_collision_id = sim.model._geom_name2id["cup_base_table_contact"]
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 == cup_base_collision_id and con.geom2 == ball_collision_id
            collision_trans = con.geom1 == ball_collision_id and con.geom2 == cup_base_collision_id

            if collision or collision_trans:
                return True
        return False

    def check_collision(self, sim):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 in self.collision_ids and con.geom2 == self.ball_collision_id
            collision_trans = con.geom1 == self.ball_collision_id and con.geom2 in self.collision_ids

            if collision or collision_trans:
                return True
        return False
