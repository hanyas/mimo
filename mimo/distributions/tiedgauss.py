from operator import add
from functools import reduce, partial

import numpy as np

from mimo.distributions import GaussianWithPrecision
from mimo.util.matrix import inv_pd


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

    def statistics(self, data, labels, keepdim=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            labels = labels[idx]

            stats = []
            for idx, c in enumerate(self.components):
                stats.append(c.statistics(data[labels == idx, :], keepdim))

            return stats
        else:
            func = partial(self.statistics, keepdim=keepdim)
            stats = list(map(func, data, labels))
            return stats if keepdim else reduce(lambda a, b: list(map(add, a, b)), stats)

    def weighted_statistics(self, data, weights, keepdim=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx, :]

            stats = []
            for idx, c in enumerate(self.components):
                stats.append(c.weighted_statistics(data, weights[:, idx], keepdim))

            return stats
        else:
            func = partial(self.weighted_statistics, keepdim=keepdim)
            stats = list(map(func, data, weights))
            return stats if keepdim else reduce(lambda a, b: list(map(add, a, b)), stats)

    # Max likelihood
    def max_likelihood(self, data, weights):
        assert weights is not None
        stats = self.weighted_statistics(data, weights)

        _sigma = np.zeros((self.dim, self.dim))
        for c, s in zip(self.components, stats):
            x, n, xxT, n = s
            c.mu = x / n
            _sigma += (xxT / n - np.outer(c.mu, c.mu))

        self.lmbda = inv_pd(_sigma / self.size)

        return self
