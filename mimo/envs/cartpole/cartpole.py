import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


def normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Cartpole(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 1

        self._dt = 0.01

        self._sigma = 1.e-4

        self._global = True

        # g = [x, th, dx, dth]
        self._goal = np.array([0., 0., 0., 0.])
        self._goal_weight = - np.array([1e0, 2e0, 1e-1, 1e-1])

        # x = [x, th, dx, dth]
        self._state_max = np.array([5., np.inf, 5., 10.])

        # x = [x, th, dx, dth]
        self._obs_max = np.array([5., np.inf, 5., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1e-2])
        self._act_max = 5.0
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
        # Equations: http://coneural.org/florian/papers/05_cart_pole.pdf
        # x = [x, th, dx, dth]
        g = 9.81
        Mc = 0.37
        Mp = 0.127
        Mt = Mc + Mp
        l = 0.3365

        def f(x, u):
            th = x[1]
            dth2 = np.power(x[3], 2)
            sth = np.sin(th)
            cth = np.cos(th)

            _num = g * sth + cth * (- u - Mp * l * dth2 * sth) / Mt
            _denom = l * ((4. / 3.) - Mp * cth**2 / Mt)
            th_acc = _num / _denom

            x_acc = (u + Mp * l * (dth2 * sth - th_acc * cth)) / Mt

            return np.hstack((x[2], x[3], x_acc, th_acc))

        c1 = f(x, u)
        c2 = f(x + 0.5 * self.dt * c1, u)
        c3 = f(x + 0.5 * self.dt * c2, u)
        c4 = f(x + self.dt * c3, u)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        return xn

    def observe(self, x):
        return np.array([x[0], normalize(x[1]), x[2], x[3]])

    def noise(self, x=None, u=None):
        return self._sigma * np.eye(self.dm_state)

    def rewrad(self, x, u):
        _x = np.array([x[0], normalize(x[1]), x[2], x[3]])
        return (_x - self._goal).T @ np.diag(self._goal_weight) @ (_x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # state-action dependent noise
        _sigma = self.noise(self.state, u)

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
            _low = np.array([-0.1, -np.pi, -5.0, -10.0])
            _high = np.array([0.1, np.pi, 5.0, 10.0])
        else:
            _low, _high = np.array([0., np.pi - np.pi / 18., 0., -1.0]),\
                          np.array([0., np.pi + np.pi / 18., 0., 1.0])

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


class CartpoleWithCartesianObservation(Cartpole):

    def __init__(self):
        super(CartpoleWithCartesianObservation, self).__init__()
        self.dm_obs = 5

        # o = [x, cos, sin, xd, thd]
        self._obs_max = np.array([5., 1., 1., 5., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

    def observe(self, x):
        return np.array([x[0],
                         np.cos(x[1]),
                         np.sin(x[1]),
                         x[2],
                         x[3]])
