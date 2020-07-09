import numpy as np
import numpy.random as npr

from operator import add
from functools import reduce, partial

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.util.data import flattendata


class GaussianWithDiagonalCovariance(Distribution):

    def __init__(self, mu=None, sigmas=None):
        self.mu = mu

        self._sigmas = sigmas
        self._sigma_chol = None

    @property
    def params(self):
        return self.mu, self.sigmas

    @params.setter
    def params(self, values):
        self.mu, self.sigmas = values

    @property
    def nb_params(self):
        return self.dim + self.dim

    @property
    def dim(self):
        return self.mu.shape[0]

    @property
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, values):
        self._sigmas = values
        self._sigma_chol = None

    @property
    def sigma(self):
        assert self._sigmas is not None
        return np.diag(self._sigmas)

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.diag(np.sqrt(self._sigmas))
        return self._sigma_chol

    @property
    def lmbdas(self):
        return 1. / self.sigmas

    @property
    def lmbda(self):
        return np.diag(self.lmbdas)

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
            xx = np.einsum('nk,nk->nk', data, data)
            n = np.ones((data.shape[0], ))

            if not vectorize:
                x = np.sum(x, axis=0)
                xx = np.sum(xx, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, n, xx])
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
            xx = np.einsum('nk,n,nk->nk', data, weights, data)
            n = weights

            if not vectorize:
                x = np.sum(x, axis=0)
                xx = np.sum(xx, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, n, xx])
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
        sigmas = - 0.5 * (1. / params[1])
        mu = - 2. * sigmas @ params[0]
        return Stats([mu, sigmas])

    @staticmethod
    def nat_to_std(natparam):
        sigmas = - 0.5 * (1. / natparam[1])
        mu = - 0.5 * sigmas * natparam[0]
        return Stats([mu, sigmas])

    def log_partition(self):
        return 0.5 * np.einsum('k,kh,h->', self.mu, self.lmbda, self.mu)\
               + 0.5 * np.sum(np.log(self.sigmas))

    def expected_statistics(self):
        E_x = self.mu
        E_xx = self.mu**2 + self.sigmas
        return E_x, E_xx

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        x, n, n, xx = stats
        self.mu = x / n
        self.sigmas = xx / n - self.mu**2

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


class GaussianWithDiagonalPrecision(Distribution):

    def __init__(self, mu=None, lmbdas=None):
        self.mu = mu

        self._lmbdas = lmbdas
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.mu, self.lmbdas

    @params.setter
    def params(self, values):
        self.mu, self.lmbdas = values

    @property
    def nb_params(self):
        return self.dim + self.dim

    @property
    def dim(self):
        return self.mu.shape[0]

    @property
    def lmbdas(self):
        return self._lmbdas

    @lmbdas.setter
    def lmbdas(self, value):
        self._lmbdas = value
        self._lmbda_chol = None

    @property
    def lmbda(self):
        assert self._lmbdas is not None
        return np.diag(self._lmbdas)

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = np.diag(np.sqrt(self._lmbdas))
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = np.diag(1. / np.sqrt(self._lmbdas))
        return self._lmbda_chol_inv

    @property
    def sigmas(self):
        return 1. / self.lmbdas

    @property
    def sigma(self):
        return np.diag(self.sigmas)

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
            xx = np.einsum('nk,nk->nk', data, data)
            n = np.ones((data.shape[0], self.dim))

            if not vectorize:
                x = np.sum(x, axis=0)
                xx = np.sum(xx, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, n, xx])
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
            xx = np.einsum('nk,n,nk->nk', data, weights, data)
            n = np.repeat(weights[:, None], self.dim, axis=1)

            if not vectorize:
                x = np.sum(x, axis=0)
                xx = np.sum(xx, axis=0)
                n = np.sum(n, axis=0)

            return Stats([x, n, n, xx])
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
        mu = params[1] * params[0]
        lmbdas = - 0.5 * params[1]
        return Stats([mu, lmbdas])

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * natparam[1] * natparam[0]
        lmbdas = - 0.5 * natparam[1]
        return Stats([mu, lmbdas])

    def log_partition(self):
        return 0.5 * np.einsum('k,kh,h->', self.mu, self.lmbda, self.mu)\
               - 0.5 * np.sum(np.log(self.lmbdas))

    def expected_statistics(self):
        E_x = self.mu
        E_xx = self.mu**2 + self.sigmas
        return E_x, E_xx

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        stats = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        x, n, n, xx = stats
        self.mu = x / n
        self.lmbdas = 1. / (xx / n - self.mu**2)

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
