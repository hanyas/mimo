import numpy as np
import numpy.random as npr


# env
def sample_env(env, nb_rollouts, nb_steps,
               ctl=None, noise_std=0.1,
               apply_limit=True):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    ulim = env.action_space.high

    for n in range(nb_rollouts):
        _obs = np.zeros((nb_steps, dm_obs))
        _act = np.zeros((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                # unifrom distribution
                u = np.random.uniform(-ulim, ulim)
            else:
                u = ctl(x)
                u = u + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act
