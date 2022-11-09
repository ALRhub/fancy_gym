import gym_blockpush
import gym

env = gym.make("blockpush-v0")
env.start()
env.scene.reset()
for i in range(100):
    env.step(env.action_space.sample())
    env.render()