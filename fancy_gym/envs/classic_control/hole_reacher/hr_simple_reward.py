import numpy as np


class HolereacherReward:
    def __init__(self, allow_self_collision, allow_wall_collision, collision_penalty):
        self.collision_penalty = collision_penalty

        # collision
        self.allow_self_collision = allow_self_collision
        self.allow_wall_collision = allow_wall_collision
        self.collision_penalty = collision_penalty
        self._is_collided = False

        self.reward_factors = np.array((-1, -5e-8, -collision_penalty))

    def reset(self):
        self._is_collided = False

    def get_reward(self, env):
        dist_cost = 0
        collision_cost = 0
        success = False

        self_collision = False
        wall_collision = False

        if not self.allow_self_collision:
            self_collision = env._check_self_collision()

        if not self.allow_wall_collision:
            wall_collision = env.check_wall_collision()

        self._is_collided = self_collision or wall_collision

        if env._steps == 199 or self._is_collided:
            # return reward only in last time step
            # Episode also terminates when colliding, hence return reward
            dist = np.linalg.norm(env.end_effector - env._goal)
            dist_cost = dist ** 2
            collision_cost = int(self._is_collided)

            success = dist < 0.005 and not self._is_collided

        info = {"is_success": success,
                "is_collided": self._is_collided,
                "end_effector": np.copy(env.end_effector)}

        acc_cost = np.sum(env._acc ** 2)

        reward_features = np.array((dist_cost, acc_cost, collision_cost))
        reward = np.dot(reward_features, self.reward_factors)

        return reward, info
