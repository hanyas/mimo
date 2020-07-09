import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg

from operator import add
from functools import reduce, partial

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.util.data import flattendata
from mimo.util.matrix import invpd


class GaussianWithCovariance(Distribution):

    def __init__(self, mu=None, sigma=None):
        self.mu = mu

        self._sigma = sigma
        self._sigma_chol = None
        self._sigma_chol_inv = None

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
        self._sigma_chol_inv = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    @property
    def sigma_chol_inv(self):
        if self._sigma_chol_inv is None:
            self._sigma_chol_inv = sc.linalg.inv(self.sigma_chol)
        return self._sigma_chol_inv

    @property
    def lmbda(self):
        return self.sigma_chol_inv.T @ self.sigma_chol_inv

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
        x = np.nan_to_num(x).reshape((-1, self.dim))

        log_lik = np.einsum('k,kh,nh->n', self.mu, self.lmbda, x)\
                  - 0.5 * np.einsum('nk,kh,nh->n', x, self.lmbda, x)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, data, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            x = data
            xxT = np.einsum('nk,nh->nkh', data, data)
            n = np.ones((data.shape[0], ))

            if not vectorize:
                x = np.sum(x, axis=0)
                xxT = np.sum(xxT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, data))
            return stats if vectorize else reduce(add, stats)

    def weighted_statistics(self, data, weights, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            x = np.einsum('n,nk->nk', weights, data)
            xxT = np.einsum('nk,n,nh->nkh', data, weights, data)
            n = weights

            if not vectorize:
                x = np.sum(x, axis=0)
                xxT = np.sum(xxT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = list(map(func, data, weights))
            return stats if vectorize else reduce(add, stats)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.dim / 2.)

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        sigma = - 0.5 * np.linalg.inv(params[1])
        mu = - 2. * sigma @ params[0]
        return Stats([mu, sigma])

    @staticmethod
    def nat_to_std(natparam):
        sigma = - 0.5 * np.linalg.inv(natparam[1])
        mu = - 0.5 * sigma @ natparam[0]
        return Stats([mu, sigma])

    def log_partition(self):
        return 0.5 * np.einsum('k,kh,h->', self.mu, self.lmbda, self.mu)\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def expected_statistics(self):
        E_x = self.mu
        E_xxT = np.outer(self.mu, self.mu) + self.sigma
        return E_x, E_xxT

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

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


class GaussianWithPrecision(Distribution):

    def __init__(self, mu=None, lmbda=None):
        self.mu = mu

        self._lmbda = lmbda
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.mu, self.lmbda

    @params.setter
    def params(self, values):
        self.mu, self.lmbda = values

    @property
    def nb_params(self):
        return self.dim + self.dim * (self.dim + 1) / 2

    @property
    def dim(self):
        return self.mu.shape[0]

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._lmbda = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def lmbda_chol(self):
        # upper cholesky triangle
        if self._lmbda_chol is None:
            self._lmbda_chol = sc.linalg.cholesky(self.lmbda, lower=False)
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = sc.linalg.inv(self.lmbda_chol)
            # self._lmbda_chol_inv = np.linalg.cholesky(np.linalg.inv(self.lmbda))  # for debugging
        return self._lmbda_chol_inv

    @property
    def sigma(self):
        return self.lmbda_chol_inv @ self.lmbda_chol_inv.T

    def rvs(self, size=1):
        if size == 1:
            return self.mu + npr.normal(size=self.dim).dot(self.lmbda_chol_inv.T)
        else:
            size = tuple([size, self.dim])
            return self.mu + npr.normal(size=size).dot(self.lmbda_chol_inv.T)

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def log_likelihood(self, x):
        bads = np.isnan(np.atleast_2d(x)).any(axis=1)
        x = np.nan_to_num(x).reshape((-1, self.dim))

        log_lik = np.einsum('k,kh,nh->n', self.mu, self.lmbda, x)\
                  - 0.5 * np.einsum('nk,kh,nh->n', x, self.lmbda, x)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, data, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            x = data
            xxT = np.einsum('nk,nh->nkh', data, data)
            n = np.ones((data.shape[0], ))

            if not vectorize:
                x = np.sum(x, axis=0)
                xxT = np.sum(xxT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, data))
            return stats if vectorize else reduce(add, stats)

    def weighted_statistics(self, data, weights, vectorize=False):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            x = np.einsum('n,nk->nk', weights, data)
            xxT = np.einsum('nk,n,nh->nkh', data, weights, data)
            n = weights

            if not vectorize:
                x = np.sum(x, axis=0)
                xxT = np.sum(xxT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = list(map(func, data, weights))
            return stats if vectorize else reduce(add, stats)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.dim / 2.)

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        mu = params[1] @ params[0]
        lmbda = - 0.5 * params[1]
        return Stats([mu, lmbda])

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * natparam[1] @ natparam[0]
        lmbda = - 0.5 * natparam[1]
        return Stats([mu, lmbda])

    def log_partition(self):
        return 0.5 * np.einsum('k,kh,h->', self.mu, self.lmbda, self.mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def expected_statistics(self):
        E_x = self.mu
        E_xxT = np.outer(self.mu, self.mu) + self.sigma
        return E_x, E_xxT

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        x, n, xxT, n = stats
        if n < self.dim or np.sum(np.linalg.svd(xxT, compute_uv=False) > 1e-6) < self.dim:
            self.mu = np.zeros(self.dim)
            self.lmbda = np.eye(self.dim)
        else:
            self.mu = x / n
            self.lmbda = invpd(xxT / n - np.outer(self.mu, self.mu))
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
