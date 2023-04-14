import fancy_gym

def compare_bases_shape(env1_id, env2_id):
    env1 = fancy_gym.make(env1_id, seed=0)
    env1.traj_gen.show_scaled_basis(plot=True)
    env2 = fancy_gym.make(env2_id, seed=0)
    env2.traj_gen.show_scaled_basis(plot=True)
    return
if __name__ == '__main__':
    compare_bases_shape("TableTennis4DProDMP-v0", "TableTennis4DProMP-v0")