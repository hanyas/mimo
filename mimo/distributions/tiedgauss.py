from operator import add
from functools import reduce, partial

import numpy as np

from mimo.abstraction import Statistics as Stats
from mimo.distributions import GaussianWithPrecision

from mimo.util.matrix import invpd


class TiedGaussiansWithPrecision:

    def __init__(self, mus, lmbda):
        self._lmbda = lmbda
        self.components = [GaussianWithPrecision(mu=_mu, lmbda=lmbda)
                           for _mu in mus]

    @property
    def params(self):
        return self.mus, self.lmbda

    @params.setter
    def params(self, values):
        self.mus, self.lmbda = values

    @property
    def nb_params(self):
        return sum(c.nb_params for c in self.components)\
               - (self.size - 1) * self.dim * (self.dim + 1) / 2

    @property
    def dim(self):
        return self.lmbda.shape[0]

    @property
    def size(self):
        return len(self.components)

    @property
    def mus(self):
        return [c.mu for c in self.components]

    @mus.setter
    def mus(self, values):
        for idx, c in enumerate(self.components):
            c.mu = values[idx]

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._lmbda = value
        for c in self.components:
            c.lmbda = value

    def statistics(self, data, labels, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            labels = labels[idx]

            stats = [c.statistics(data[labels == idx, :], vectorize)
                     for idx, c in enumerate(self.components)]

            return Stats(stats)
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, data, labels))
            return list(stats) if vectorize else reduce(add, stats)

    def weighted_statistics(self, data, weights, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            stats = [c.weighted_statistics(data, weights[:, idx], vectorize)
                     for idx, c in enumerate(self.components)]

            return Stats(stats)
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = map(func, data, weights)
            return list(stats) if vectorize else reduce(add, stats)

    # Max likelihood
    def max_likelihood(self, data, weights):
        assert weights is not None
        stats = self.weighted_statistics(data, weights)

        _n = 0
        _sigma = np.zeros((self.dim, self.dim))
        for c, s in zip(self.components, stats):
            x, n, xxT, n = s
            c.mu = x / n

            _n += n
            _sigma += xxT - n * np.outer(c.mu, c.mu)

        self.lmbda = invpd(_sigma / _n)

        return self
