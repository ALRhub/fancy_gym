import numpy as np
from alr_envs.mujoco import alr_reward_fct


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
        self.dists = []
        self.dists_ctxt = []
        self.dists_final = []
        self.costs = []
        self.context = context

    def compute_reward(self, action, sim, step):
        self.ball_id = sim.model._body_name2id["ball"]
        self.ball_collision_id = sim.model._geom_name2id["ball_geom"]
        self.goal_id = sim.model._site_name2id["cup_goal"]
        self.goal_final_id = sim.model._site_name2id["cup_goal_final"]
        self.collision_ids = [sim.model._geom_name2id[name] for name in self.collision_objects]

        ball_in_cup = self.check_ball_in_cup(sim, self.ball_collision_id)

        # Compute the current distance from the ball to the inner part of the cup
        goal_pos = sim.data.site_xpos[self.goal_id]
        ball_pos = sim.data.body_xpos[self.ball_id]
        goal_final_pos = sim.data.site_xpos[self.goal_final_id]
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_ctxt.append(np.linalg.norm(ball_pos - self.context))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.ball_traj[step, :] = ball_pos

        action_cost = np.sum(np.square(action))

        stop_sim = False
        success = False

        if self.check_collision(sim):
            reward = - 1e-5 * action_cost - 1000
            stop_sim = True
            return reward, success, stop_sim

        if ball_in_cup or step == self.sim_time - 1:
            min_dist = np.min(self.dists)
            dist_final = self.dists_final[-1]
            dist_ctxt = self.dists_ctxt[-1]

            cost = self._get_stage_wise_cost(ball_in_cup, min_dist, dist_final, dist_ctxt)
            reward = np.exp(-1 * cost) - 1e-5 * action_cost
            stop_sim = True
            success = dist_final < 0.05 and ball_in_cup
        else:
            reward = - 1e-5 * action_cost
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
