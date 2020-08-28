import gym


if __name__ == "__main__":
    env = gym.make('reacher:ReacherALREnv-v0')
    #env = gym.make('Hopper-v2')
    env.reset()

    for i in range(10000):
        action = env.action_space.sample()
        obs = env.step(action)
        print("step",i)
        env.render()
