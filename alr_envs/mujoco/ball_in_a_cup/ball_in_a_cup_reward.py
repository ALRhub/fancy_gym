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
        self.dists_final = None
        self.costs = None

        self.reset(None)

    def reset(self, context):
        self.ball_traj = np.zeros(shape=(self.sim_time, 3))
        self.dists = []
        self.dists_final = []
        self.costs = []
        self.context = context

    def compute_reward(self, action, sim, step, context=None):
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
        self.ball_traj[step, :] = ball_pos

        action_cost = np.sum(np.square(action))

        if self.check_collision(sim):
            reward = - 1e-5 * action_cost - 1000
            return reward, False, True

        if step == self.sim_time - 1:
            min_dist = np.min(self.dists)
            dist_final = self.dists_final[-1]

            cost = 0.5 * min_dist + 0.5 * dist_final
            reward = np.exp(-2 * cost) - 1e-5 * action_cost
            success = dist_final < 0.05 and ball_in_cup
        else:
            reward = - 1e-5 * action_cost
            success = False

        return reward, success, False

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
