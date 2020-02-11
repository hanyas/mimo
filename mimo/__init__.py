from gym.envs.registration import register


register(
    id='BouncingBall-DPGLM-v0',
    entry_point='mimo.envs:BouncingBall',
    max_episode_steps=1000,
)

register(
    id='Pendulum-DPGLM-v0',
    entry_point='mimo.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='Pendulum-DPGLM-v1',
    entry_point='mimo.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='Cartpole-DPGLM-v0',
    entry_point='mimo.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-DPGLM-v1',
    entry_point='mimo.envs:CartpoleWithCartesianObservation',
    max_episode_steps=1000,
)
