from gym.envs.registration import register

register(
    id='ReacherALREnv-v0',
    entry_point='reacher.envs:ReacherALREnv',
)
