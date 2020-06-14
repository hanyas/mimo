from functools import reduce
from operator import add

import numpy as np

from mimo.abstraction import Statistics as Stats
from mimo.distributions import Gaussian


class TiedGaussians:

    def __init__(self, mus, sigma):
        self.components = [Gaussian(mu=_mu, sigma=sigma) for _mu in mus]

    @property
    def params(self):
        return self.mus, self.sigma

    @params.setter
    def params(self, values):
        self.mus, self.sigma = values

    @property
    def mus(self):
        return [c.mu for c in self.components]

    @mus.setter
    def mus(self, values):
        for idx, c in enumerate(self.components):
            c.mu = values[idx]

    @property
    def sigma(self):
        assert np.all([c.sigma == self.components[0].sigma
                       for c in self.components])
        return self.components[0].sigma

    @sigma.setter
    def sigma(self, value):
        for c in self.components:
            c.sigma = value

    @property
    def size(self):
        return len(self.components)

    @property
    def dim(self):
        return self.components[0].dim

    @property
    def nb_params(self):
        return sum(c.nb_params for c in self.components)\
               - (self.size - 1) * self.dim * (self.dim + 1) / 2

    def get_statistics(self, data, labels):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            labels = labels[idx]

            stats = []
            for k in range(self.size):
                _data = data[labels == k, :]
                n = _data.shape[0]
                x = np.sum(_data, axis=0)
                xxT = np.einsum('ni,nj->ij', _data, _data)

                stats.append(Stats([x, n, xxT, n]))

            return stats
        else:
            return reduce(lambda a, b: list(map(add, a, b)),
                          list(map(self.get_statistics, data, labels)))

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx, :]

            stats = []
            for k in range(self.size):
                n = np.sum(weights[:, k], axis=0)
                x = np.einsum('ni,n->i', data, weights[:, k])
                xxT = np.einsum('ni,n,nj->ij', data, weights[:, k], data)

                stats.append(Stats([x, n, xxT, n]))

            return stats
        else:
            return reduce(lambda a, b: list(map(add, a, b)),
                          list(map(self.get_weighted_statistics, data, weights)))

    def _empty_statistics(self):
        return Stats([c._empty_statsitics() for c in self.components])

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)

        _sigma = []
        for c, s in zip(self.components, stats):
            x, n, xxT, n = s
            c.mu = x / n

            uut = n * np.outer(c.mu, c.mu)
            _sigma.append((xxT - uut) / n)

        self.sigma = np.mean(np.stack(_sigma, axis=2), axis=-1)

        return self
