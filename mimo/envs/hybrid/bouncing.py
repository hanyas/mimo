import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class BouncingBall(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.01

        self._sigma = 1e-8

        # x = [x, xd]
        self._xmax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self._umax = 0.
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self._xmax

    @property
    def ulim(self):
        return self._umax

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return NotImplementedError

    def dynamics(self, x, u):
        k, g = 0.8, 9.81

        def f(x, u):
            h, dh = x
            return np.array([dh, -g])

        c1 = f(x, u)
        c2 = f(x + 0.5 * self.dt * c1, u)
        c3 = f(x + 0.5 * self.dt * c2, u)
        c4 = f(x + self.dt * c3, u)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        if xn[0] <= 0. and xn[1] < 0.:
            xn[0] = 0.
            xn[1] = - k * xn[1]

        return xn

    def observe(self, x):
        return x

    def noise(self, x=None, u=None):
        return self._sigma * np.eye(self.dm_state)

    def rewrad(self, x, u):
        return NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # apply action constraints
        _u = np.clip(u, -self.ulim, self.ulim)

        # state-action dependent noise
        _sigma = self.noise(self.state, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(self.state, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self.xlim, self.xlim)

        # compute reward
        rwrd = self.rewrad(self.state, _u)

        # add noise
        self.state = self.np_random.multivariate_normal(mean=_xn, cov=_sigma)

        return self.observe(self.state), rwrd, False, {}

    # following functions for plotting
    def fake_step(self, x, u):
        # apply action constraints
        _u = np.clip(u, -self.ulim, self.ulim)

        # state-action dependent noise
        _sigma = self.noise(x, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(x, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self.xlim, self.xlim)

        return self.observe(_xn)

    def reset(self):
        _low = np.array([10.0, -2.0])
        _high = np.array([15.0, 2.0])
        self.state = self.np_random.uniform(low=_low, high=_high)
        return self.observe(self.state)
