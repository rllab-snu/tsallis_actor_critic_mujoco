from gym.envs.registration import register

register(
    id='CustomMountainCarContinuous-v0',
    entry_point='custom_gym.envs:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0
)

register(
    id='CustomPendulum-v0',
    entry_point='custom_gym.envs:PendulumEnv',
    max_episode_steps=200,
)
