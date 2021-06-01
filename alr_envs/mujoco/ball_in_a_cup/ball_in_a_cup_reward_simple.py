import numpy as np
from alr_envs.mujoco import alr_reward_fct


class BallInACupReward(alr_reward_fct.AlrReward):
    def __init__(self, sim_time):
        self.sim_time = sim_time

        self.collision_objects = ["cup_geom1", "cup_geom2", "cup_base_contact_below",
                                  "wrist_palm_link_convex_geom",
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
        self._is_collided = False
        self.collision_penalty = 1

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
        self.action_costs = []
        self.angle_costs = []
        self.cup_angles = []

    def compute_reward(self, action, env):
        self.ball_id = env.sim.model._body_name2id["ball"]
        self.ball_collision_id = env.sim.model._geom_name2id["ball_geom"]
        self.goal_id = env.sim.model._site_name2id["cup_goal"]
        self.goal_final_id = env.sim.model._site_name2id["cup_goal_final"]
        self.collision_ids = [env.sim.model._geom_name2id[name] for name in self.collision_objects]

        ball_in_cup = self.check_ball_in_cup(env.sim, self.ball_collision_id)

        # Compute the current distance from the ball to the inner part of the cup
        goal_pos = env.sim.data.site_xpos[self.goal_id]
        ball_pos = env.sim.data.body_xpos[self.ball_id]
        goal_final_pos = env.sim.data.site_xpos[self.goal_final_id]
        self.dists.append(np.linalg.norm(goal_pos - ball_pos))
        self.dists_final.append(np.linalg.norm(goal_final_pos - ball_pos))
        self.ball_traj[env._steps, :] = ball_pos
        cup_quat = np.copy(env.sim.data.body_xquat[env.sim.model._body_name2id["cup"]])
        cup_angle = np.arctan2(2 * (cup_quat[0] * cup_quat[1] + cup_quat[2] * cup_quat[3]),
                                          1 - 2 * (cup_quat[1]**2 + cup_quat[2]**2))
        cost_angle = (cup_angle - np.pi / 2) ** 2
        self.angle_costs.append(cost_angle)
        self.cup_angles.append(cup_angle)

        action_cost = np.sum(np.square(action))
        self.action_costs.append(action_cost)

        self._is_collided = self.check_collision(env.sim) or env.check_traj_in_joint_limits()

        if env._steps == env.sim_steps - 1 or self._is_collided:
            t_min_dist = np.argmin(self.dists)
            angle_min_dist = self.cup_angles[t_min_dist]
            # cost_angle = (angle_min_dist - np.pi / 2)**2


            min_dist = self.dists[t_min_dist]
            dist_final = self.dists_final[-1]
            min_dist_final = np.min(self.dists_final)

            cost = 0.5 * dist_final + 0.05 * cost_angle  # TODO: Increase cost_angle weight  # 0.5 * min_dist +
            # reward = np.exp(-2 * cost) - 1e-2 * action_cost - self.collision_penalty * int(self._is_collided)
            # reward = - dist_final**2 - 1e-4 * cost_angle - 1e-5 * action_cost - self.collision_penalty * int(self._is_collided)
            reward = - dist_final**2 - min_dist_final**2 - 1e-4 * cost_angle - 5e-4 * action_cost - self.collision_penalty * int(self._is_collided)
            success = dist_final < 0.05 and ball_in_cup and not self._is_collided
            crash = self._is_collided
        else:
            reward = - 5e-4 * action_cost - 1e-4 * cost_angle  # TODO: increase action_cost weight
            success = False
            crash = False

        return reward, success, crash

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
