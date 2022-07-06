import numpy as np
from alr_envs.alr.mujoco.ball_in_a_cup import alr_reward_fct


class BallInACupReward(alr_reward_fct.AlrReward):
    def __init__(self, sim_time):
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
        self.cup_traj = np.zeros(shape=(self.sim_time, 3))
        self.dists = []
        self.dists_ctxt = []
        self.dists_final = []
        self.costs = []
        self.context = context
        self.ball_in_cup = False
        self.ball_above_threshold = False
        self.dist_ctxt = 3
        self.action_costs = []
        self.cup_angles = []

    def compute_reward(self, action, sim, step):
        action_cost = np.sum(np.square(action))
        self.action_costs.append(action_cost)

        stop_sim = False
        success = False

        self.ball_id = sim.model._body_name2id["ball"]
        self.ball_collision_id = sim.model._geom_name2id["ball_geom"]
        self.goal_id = sim.model._site_name2id["cup_goal"]
        self.goal_final_id = sim.model._site_name2id["cup_goal_final"]
        self.collision_ids = [sim.model._geom_name2id[name] for name in self.collision_objects]

        if self.check_collision(sim):
            reward = - 1e-3 * action_cost - 1000
            stop_sim = True
            return reward, success, stop_sim

        # Compute the current distance from the ball to the inner part of the cup
        goal_pos = sim.data.site_xpos[self.goal_id]
        ball_pos = sim.data.body_xpos[self.ball_id]
        goal_final_pos = sim.data.site_xpos[self.goal_final_id]
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.dists_ctxt.append(np.linalg.norm(ball_pos - self.context))
        self.ball_traj[step, :] = np.copy(ball_pos)
        self.cup_traj[step, :] = np.copy(goal_pos)  # ?
        cup_quat = np.copy(sim.data.body_xquat[sim.model._body_name2id["cup"]])
        self.cup_angles.append(np.arctan2(2 * (cup_quat[0] * cup_quat[1] + cup_quat[2] * cup_quat[3]),
                                          1 - 2 * (cup_quat[1] ** 2 + cup_quat[2] ** 2)))

        # Determine the first time when ball is in cup
        if not self.ball_in_cup:
            ball_in_cup = self.check_ball_in_cup(sim, self.ball_collision_id)
            self.ball_in_cup = ball_in_cup
            if ball_in_cup:
                dist_to_ctxt = np.linalg.norm(ball_pos - self.context)
                self.dist_ctxt = dist_to_ctxt

        if step == self.sim_time - 1:
            t_min_dist = np.argmin(self.dists)
            angle_min_dist = self.cup_angles[t_min_dist]
            cost_angle = (angle_min_dist - np.pi / 2) ** 2

            min_dist = np.min(self.dists)
            dist_final = self.dists_final[-1]
            # dist_ctxt = self.dists_ctxt[-1]

            #  # max distance between ball and cup and cup height at that time
            # ball_to_cup_diff = self.ball_traj[:, 2] - self.cup_traj[:, 2]
            # t_max_diff = np.argmax(ball_to_cup_diff)
            # t_max_ball_height = np.argmax(self.ball_traj[:, 2])
            # max_ball_height = np.max(self.ball_traj[:, 2])

            # cost = self._get_stage_wise_cost(ball_in_cup, min_dist, dist_final, dist_ctxt)
            cost = 0.5 * min_dist + 0.5 * dist_final + 0.3 * np.minimum(self.dist_ctxt, 3) + 0.01 * cost_angle
            reward = np.exp(-2 * cost) - 1e-3 * action_cost
            # if max_ball_height < self.context[2] or ball_to_cup_diff[t_max_ball_height] < 0:
            #     reward -= 1

            success = dist_final < 0.05 and self.dist_ctxt < 0.05
        else:
            reward = - 1e-3 * action_cost
            success = False

        return reward, success, stop_sim

    def _get_stage_wise_cost(self, ball_in_cup, min_dist, dist_final, dist_to_ctxt):
        if not ball_in_cup:
            cost = 3 + 2*(0.5 * min_dist**2 + 0.5 * dist_final**2)
        else:
            cost = 2 * dist_to_ctxt ** 2
            print('Context Distance:', dist_to_ctxt)
        return cost

    def check_ball_in_cup(self, sim, ball_collision_id):
        cup_base_collision_id = sim.model._geom_name2id["cup_base_contact"]
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
