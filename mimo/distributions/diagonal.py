from functools import reduce

import numpy as np

from mimo.abstraction import Statistics as Stats
from mimo.distributions import Gaussian


class DiagonalGaussian(Gaussian):

    def __init__(self, mu, sigmas=None):
        self._sigmas = sigmas
        super(DiagonalGaussian, self).__init__(mu=mu, sigma=self.sigma)

    @property
    def nb_params(self):
        return self.dim + self.dim

    @property
    def sigma(self):
        if self._sigmas is not None:
            return np.diag(self._sigmas)
        else:
            return None

    @sigma.setter
    def sigma(self, value):
        value = np.array(value)
        if value.ndim == 1:
            self._sigmas = value
        else:
            self._sigmas = np.diag(value)
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.diag(np.sqrt(self._sigmas))
        return self._sigma_chol

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            xx = np.einsum('nk,nk->k', data, data)
            x = np.sum(data, axis=0)
            n = np.repeat(data.shape[0], self.dim)
            return Stats([x, n, n, xx])
        else:
            return reduce(lambda a, b: a + b,
                          list(map(self.get_statistics, data)))

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            xx = np.einsum('nk,n,nk->k', data, weights, data)
            x = weights.dot(data)
            n = np.repeat(weights.sum(), self.dim)
            return Stats([x, n, n, xx])
        else:
            return reduce(lambda a, b: a + b,
                          list(map(self.get_weighted_statistics, data, weights)))

    def _empty_statistics(self):
        return Stats([np.zeros((self.dim, )), np.zeros((self.dim, )),
                      np.zeros((self.dim, )), np.zeros((self.dim, ))])

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.posterior.get_statistics(data) if weights is None\
            else self.posterior.get_weighted_statistics(data, weights)

        x, n, n, xx = stats
        self.mu = x / n
        self.sigma = xx / n - self.mu**2

        return self
