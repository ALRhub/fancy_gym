import gymnasium as gym
import fancy_gym


def compare_bases_shape(env1_id, env2_id):
    env1 = gym.make(env1_id)
    env1.traj_gen.show_scaled_basis(plot=True)
    env2 = gym.make(env2_id)
    env2.traj_gen.show_scaled_basis(plot=True)
    return


if __name__ == '__main__':
    compare_bases_shape("fancy_ProDMP/TableTennis4D-v0", "fancy_ProMP/TableTennis4D-v0")
