import numpy as np


class BallInACupReward:
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

        self.ctxt_id = None
        self.ball_id = None
        self.ball_collision_id = None
        self.goal_id = None
        self.goal_final_id = None
        self.collision_ids = None

        self.ball_traj = None
        self.dists = None
        self.dists_final = None
        self.costs = None

        self.reset()

    def reset(self):
        self.ball_traj = np.zeros(shape=(self.sim_time, 3))
        self.dists = []
        self.dists_final = []
        self.costs = []

    def compute_reward(self, action, sim, step):
        self.ctxt_id = sim.model._site_name2id['context_point']
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
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        # dists_ctxt.append(np.linalg.norm(ball_pos - ctxt))
        self.ball_traj[step, :] = ball_pos

        if self.check_collision(sim):
            return -1000, False, True

        # self._get_cost(ball_pos, goal_pos, goal_final_pos, action,
        #                sim.data.get_site_xpos('context_point').copy(), step)

        # min_dist = np.min(self.dists)
        # dist_final = self.dists_final[-1]
        action_cost = np.sum(np.square(action))

        # cost = self.get_stage_wise_cost(ball_in_cup, min_dist, self.dists_final[-1])  # , self.dists_ctxt[-1])
        if step == self.sim_time - 1:
            min_dist = np.min(self.dists)
            dist_final = self.dists_final[-1]

            cost = 0.5 * min_dist + 0.5 * dist_final
            # cost = 3 + 2 * (0.5 * min_dist ** 2 + 0.5 * dist_final ** 2)
            reward = np.exp(-2 * min_dist) - 1e-5 * action_cost
            success = dist_final < 0.05 and min_dist < 0.05
        else:
            cost = 0
            reward = - 1e-5 * action_cost
            success = False
        # action_cost = np.mean(np.sum(np.square(torques), axis=1), axis=0)

        return reward, success, False

    def get_stage_wise_cost(self, ball_in_cup, min_dist, dist_final):  #, dist_to_ctxt):
        # stop_sim = False
        cost = 3 + 2 * (0.5 * min_dist ** 2 + 0.5 * dist_final ** 2)
        # if not ball_in_cup:
        #     # cost = 3 + 2*(0.5 * min_dist + 0.5 * dist_final)
        #     cost = 3 + 2*(0.5 * min_dist**2 + 0.5 * dist_final**2)
        # else:
        #     # cost = 2*dist_to_ctxt
        #     cost = 2*dist_to_ctxt**2
        #     stop_sim = True
        #     # print(dist_to_ctxt-0.02)
        #     print('Context Distance:', dist_to_ctxt)
        return cost

    def _get_cost(self, ball_pos, goal_pos, goal_pos_final, u, ctxt, t):

        cost = 0
        if t == self.sim_time*0.8:
            dist = 0.5*np.linalg.norm(goal_pos-ball_pos)**2 + 0.5*np.linalg.norm(goal_pos_final-ball_pos)**2
            # dist_ctxt = np.linalg.norm(ctxt-goal_pos)**2
            cost = dist  # +dist_ctxt
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
