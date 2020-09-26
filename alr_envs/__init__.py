from gym.envs.registration import register

register(
    id='ALRReacher-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 5,
    }
)

register(
    id='ALRReacherShortSparse-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=50,
    kwargs={
        "steps_before_reward": 50,
        "n_links": 5,
    }
)

register(
    id='ALRReacherShort-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=50,
    kwargs={
        "steps_before_reward": 40,
        "n_links": 5,
    }
)

register(
    id='ALRReacherSparse-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 200,
        "n_links": 5,
    }
)

register(
    id='ALRReacher100-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 100,
        "n_links": 5,
    }
)

register(
    id='ALRReacher180-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 180,
        "n_links": 5,
    }
)

register(
    id='ALRReacher7-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 0,
        "n_links": 7,
    }
)

register(
    id='ALRReacher100_7-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 100,
        "n_links": 7,
    }
)

register(
    id='ALRReacher180_7-v0',
    entry_point='alr_envs.mujoco:ALRReacherEnv',
    max_episode_steps=200,
    kwargs={
        "steps_before_reward": 180,
        "n_links": 7,
    }
)

register(
    id='SimpleReacher-v0',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='SimpleReacher5-v0',
    entry_point='alr_envs.classic_control:SimpleReacherEnv',
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)
