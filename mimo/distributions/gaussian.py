import numpy as np
import numpy.random as npr

from functools import reduce

from numpy.core.umath_tests import inner1d
from scipy import linalg

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.util.data import flattendata


class Gaussian(Distribution):

    def __init__(self, mu=None, sigma=None):
        self.mu = mu

        self._sigma = sigma
        self._sigma_chol = None
        self._sigma_inv = None

    @property
    def params(self):
        return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma = values

    @property
    def nb_params(self):
        return self.dim + self.dim * (self.dim + 1) / 2

    @property
    def dim(self):
        return self.mu.shape[0]

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._sigma_chol = None
        self._sigma_inv = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            # self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    @property
    def sigma_inv(self):
        if self._sigma_inv is None:
            # self._sigma_inv = np.linalg.inv(near_pd(self.sigma))
            self._sigma_inv = np.linalg.inv(self.sigma)
        return self._sigma_inv

    def rvs(self, size=1):
        if size == 1:
            return self.mu + npr.normal(size=self.dim).dot(self.sigma_chol.T)
        else:
            size = tuple([size, self.dim])
            return self.mu + npr.normal(size=size).dot(self.sigma_chol.T)

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def log_likelihood(self, x):
        bads = np.isnan(np.atleast_2d(x)).any(axis=1)
        xc = np.nan_to_num(x).reshape((-1, self.dim)) - self.mu
        xs = linalg.solve_triangular(self.sigma_chol, xc.T, lower=True)
        out = - 0.5 * self.dim * np.log(2. * np.pi)\
              - np.sum(np.log(np.diag(self.sigma_chol))) - 0.5 * inner1d(xs.T, xs.T)
        out[bads] = 0
        return out

    def log_partition(self):
        return 0.5 * self.dim * np.log(2. * np.pi)\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * self.dim * np.log(2. * np.pi) + self.dim\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        sigma = np.linalg.inv(params[1])
        mu = sigma @ params[0]
        return Stats([mu, sigma])

    @staticmethod
    def nat_to_std(natparam):
        sigma = np.linalg.inv(natparam[1])
        mu = sigma @ natparam[0]
        return mu, sigma

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            xxT = np.einsum('nk,nh->kh', data, data)
            x = np.sum(data, axis=0)
            n = data.shape[0]
            return Stats([x, n, xxT, n])
        else:
            return reduce(lambda a, b: a + b, list(map(self.get_statistics, data)))

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            xxT = np.einsum('nk,n,nh->kh', data, weights, data)
            x = weights.dot(data)
            n = weights.sum()
            return Stats([x, n, xxT, n])
        else:
            return reduce(lambda a, b: a + b, list(map(self.get_weighted_statistics, data, weights)))

    def _empty_statistics(self):
        return Stats([np.zeros((self.dim, )), 0.,
                      np.zeros((self.dim, self.dim)), 0.])

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)

        x, n, xxT, n = stats
        if n < self.dim or np.sum(np.linalg.svd(xxT, compute_uv=False) > 1e-6) < self.dim:
            self.mu = np.zeros(self.dim)
            self.sigma = np.eye(self.dim)
        else:
            self.mu = x / n
            self.sigma = xxT / n - np.outer(self.mu, self.mu)
        return self

    def plot(self, ax=None, data=None, color='b', label='',
             alpha=1., update=False, draw=True):

        import matplotlib.pyplot as plt
        from mimo.util.plot import plot_gaussian

        ax = ax if ax else plt.gca()

        _scatterplot = None
        if data is not None:
            data = flattendata(data)
            _scatterplot = ax.scatter(data[:, 0], data[:, 1], marker='.', color=color)

        _parameterplot = plot_gaussian(self.mu, self.sigma, color=color, label=label,
                                       alpha=min(1 - 1e-3, alpha), ax=ax,
                                       artists=self._parameterplot if update else None)
        if draw:
            plt.draw()

        return [_scatterplot] + list(_parameterplot)


class GaussianWithFixedCovariance(Gaussian):

    def __init__(self, mu=None, sigma=None):
        super(GaussianWithFixedCovariance, self).__init__(mu=mu, sigma=sigma)

    @property
    def params(self):
        return self.mu

    @params.setter
    def params(self, values):
        self.mu = values

    @property
    def nb_params(self):
        return self.dim

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)

        x, n, _, _ = stats
        if n < self.dim:
            self.mu = np.zeros(self.dim)
        else:
            self.mu = x / n
        return self
