import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


def normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Pendulum(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.01

        self._sigma = 1e-8

        self._global = True

        # g = [th, thd]
        self._goal = np.array([0., 0.])
        self._goal_weight = - np.array([1e0, 1e-1])

        # x = [th, thd]
        self._state_max = np.array([np.inf, 8.0])

        # o = [th, thd]
        self._obs_max = np.array([np.inf, 8.0])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1e-3])
        self._act_max = 2.5
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self._state_max

    @property
    def ulim(self):
        return self._act_max

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return self._goal

    def dynamics(self, x, u):
        g, m, l, k = 10., 1., 1., 1e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, 3. * g / (2. * l) * np.sin(th) +
                              3. / (m * l ** 2) * (u - k * dth)))
        c1 = f(x, u)
        c2 = f(x + 0.5 * self.dt * c1, u)
        c3 = f(x + 0.5 * self.dt * c2, u)
        c4 = f(x + self.dt * c3, u)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        return xn

    def observe(self, x):
        return np.array([normalize(x[0]), x[1]])

    def noise(self, x=None, u=None):
        return self._sigma * np.eye(self.dm_state)

    def rewrad(self, x, u):
        _x = np.array([normalize(x[0]), x[1]])
        return (_x - self._goal).T @ np.diag(self._goal_weight) @ (_x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # state-action dependent noise
        _sigma = self.noise(self.state, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(self.state, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self._state_max, self._state_max)

        # compute reward
        rwrd = self.rewrad(self.state, _u)

        # add noise
        self.state = self.np_random.multivariate_normal(mean=_xn, cov=_sigma)

        return self.observe(self.state), rwrd, False, {}

    def reset(self):
        if self._global:
            _low = np.array([-np.pi, -8.0])
            _high = np.array([np.pi, 8.0])
        else:
            _low, _high = np.array([np.pi - np.pi / 18., -1.0]),\
                          np.array([np.pi + np.pi / 18., 1.0])

        self.state = self.np_random.uniform(low=_low, high=_high)
        return self.observe(self.state)

    # following functions for plotting
    def fake_step(self, x, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # state-action dependent noise
        _sigma = self.noise(x, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(x, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self._state_max, self._state_max)

        return self.observe(_xn)


class PendulumWithCartesianObservation(Pendulum):

    def __init__(self):
        super(PendulumWithCartesianObservation, self).__init__()
        self.dm_obs = 3

        # o = [cos, sin, thd]
        self._obs_max = np.array([1., 1., 8.0])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

    def observe(self, x):
        return np.array([np.cos(x[0]),
                         np.sin(x[0]),
                         x[1]])
