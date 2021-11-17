import numpy as np
from alr_envs.alr.classic_control.utils import check_self_collision


class HolereacherSimpleReward:
    def __init__(self, allow_self_collision, allow_wall_collision, collision_penalty):
        self.collision_penalty = collision_penalty

        # collision
        self.allow_self_collision = allow_self_collision
        self.allow_wall_collision = allow_wall_collision
        self.collision_penalty = collision_penalty
        self._is_collided = False
        pass

    def get_reward(self, env, action):
        reward = 0
        success = False

        self_collision = False
        wall_collision = False

        # joints = np.hstack((env._joints[:-1, :], env._joints[1:, :]))
        if not self.allow_self_collision:
            self_collision = env._check_self_collision()

        if not self.allow_wall_collision:
            wall_collision = env.check_wall_collision()

        self._is_collided = self_collision or wall_collision

        if env._steps == 199 or self._is_collided:
            # return reward only in last time step
            # Episode also terminates when colliding, hence return reward
            dist = np.linalg.norm(env.end_effector - env._goal)

            success = dist < 0.005 and not self._is_collided
            reward = - dist ** 2 - self.collision_penalty * self._is_collided

        info = {"is_success": success,
                "is_collided": self._is_collided}

        acc = (action - env._angle_velocity) / env.dt
        reward -= 5e-8 * np.sum(acc ** 2)

        return reward, info


