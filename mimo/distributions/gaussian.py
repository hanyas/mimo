from abc import ABC

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg

from operator import add
from functools import reduce, partial

from mimo.utils.abstraction import Statistics as Stats
from mimo.utils.matrix import symmetrize


class _GaussianBase:
    def __init__(self, dim, mu=None):
        self.dim = dim

        self.mu = mu

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    @property
    def sigma(self):
        raise NotImplementedError

    @sigma.setter
    def sigma(self, value):
        raise NotImplementedError

    @property
    def lmbda(self):
        raise NotImplementedError

    @lmbda.setter
    def lmbda(self, value):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        raise NotImplementedError

    @staticmethod
    def nat_to_std(natparam):
        raise NotImplementedError

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    @property
    def base(self):
        return np.power(2. * np.pi, - self.dim / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        raise NotImplementedError

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x, copy=False).reshape((-1, self.dim))

            log_lik = np.einsum('d,dl,nl->n', self.mu, self.lmbda, x, optimize=True)\
                      - 0.5 * np.einsum('nd,dl,nl->n', x, self.lmbda, x, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition() + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x))

    def expected_statistics(self):
        raise NotImplementedError

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def plot(self, ax=None, data=None, color='b',
             label='', alpha=1., draw=True):

        import matplotlib.pyplot as plt
        from mimo.utils.plot import plot_gaussian

        ax = ax if ax else plt.gca()

        scatterplot = None
        if data is not None:
            scatterplot = ax.scatter(data[:, 0], data[:, 1], marker='.', color=color)
        parameterplot = plot_gaussian(self.mu, self.sigma, color=color, label=label,
                                      alpha=min(1 - 1e-3, alpha), ax=ax, artists=None)
        if draw:
            plt.draw()

        return [scatterplot] + list(parameterplot)


class GaussianWithCovariance(_GaussianBase):

    def __init__(self, dim, mu=None, sigma=None):

        self._sigma = sigma
        self._sigma_chol = None
        self._sigma_chol_inv = None

        super(GaussianWithCovariance, self).__init__(dim, mu)

    @property
    def params(self):
        return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma = values

    @property
    def nb_params(self):
        return self.dim + self.dim * (self.dim + 1) / 2

    @staticmethod
    def std_to_nat(params):
        a = np.linalg.inv(params[1]) @ params[0]
        b = - 0.5 * np.linalg.inv(params[1])
        return Stats([a, b])

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * np.linalg.inv(natparam[1]) @ natparam[0]
        sigma = - 0.5 * np.linalg.inv(natparam[1])
        return mu, sigma

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
        size = self.dim if size == 1 else tuple([size, self.dim])
        return self.mu + npr.normal(size=size).dot(self.sigma_chol.T)

    def statistics(self, data, fold=True):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            if fold:
                c0, c1 = 'nd->d', 'nd,nl->dl'
                n = data.shape[0]
            else:
                c0, c1 = 'nd->nd', 'nd,nl->ndl'
                n = np.ones((data.shape[0], ))

            x = np.einsum(c0, data, optimize=True)
            xxT = np.einsum(c1, data, data, optimize=True)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, data))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[idx]

            c0, c1 = 'n,nd->d', 'nd,n,nl->dl'

            x = np.einsum(c0, weights, data, optimize=True)
            xxT = np.einsum(c1, data, weights, data, optimize=True)
            n = np.sum(weights)

            return Stats([x, n, xxT, n])
        else:
            stats = list(map(self.weighted_statistics, data, weights))
            return reduce(add, stats)

    def log_partition(self):
        return 0.5 * np.einsum('d,dl,l->', self.mu, self.lmbda, self.mu)\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def expected_statistics(self):
        E_x = self.mu
        E_xxT = np.outer(self.mu, self.mu) + self.sigma
        return E_x, E_xxT

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        x, n, xxT, n = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        self.mu = x / n
        sigma = xxT / n - np.outer(self.mu, self.mu)

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.sigma = sigma


class GaussianWithPrecision(_GaussianBase):

    def __init__(self, dim, mu=None, lmbda=None):

        self._lmbda = lmbda
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        super(GaussianWithPrecision, self).__init__(dim, mu)

    @property
    def params(self):
        return self.mu, self.lmbda

    @params.setter
    def params(self, values):
        self.mu, self.lmbda = values

    @property
    def nb_params(self):
        return self.dim + self.dim * (self.dim + 1) / 2

    @staticmethod
    def std_to_nat(params):
        a = params[1] @ params[0]
        b = - 0.5 * params[1]
        return Stats([a, b])

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * np.linalg.inv(natparam[1]) @ natparam[0]
        lmbda = - 2. * natparam[1]
        return mu, lmbda

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
        if self._lmbda_chol is None:
            self._lmbda_chol = sc.linalg.cholesky(self.lmbda, lower=False)
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = sc.linalg.inv(self.lmbda_chol)
        return self._lmbda_chol_inv

    @property
    def sigma(self):
        return self.lmbda_chol_inv @ self.lmbda_chol_inv.T

    def rvs(self, size=1):
        size = self.dim if size == 1 else tuple([size, self.dim])
        return self.mu + npr.normal(size=size).dot(self.lmbda_chol_inv.T)

    def statistics(self, data, fold=True):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            if fold:
                c0, c1 = 'nd->d', 'nd,nl->dl'
                n = data.shape[0]
            else:
                c0, c1 = 'nd->nd', 'nd,nl->ndl'
                n = np.ones((data.shape[0], ))

            x = np.einsum(c0, data, optimize=True)
            xxT = np.einsum(c1, data, data, optimize=True)

            return Stats([x, n, xxT, n])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, data))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[idx]

            c0, c1 = 'n,nd->d', 'nd,n,nl->dl'

            x = np.einsum(c0, weights, data, optimize=True)
            xxT = np.einsum(c1, data, weights, data, optimize=True)
            n = np.sum(weights)

            return Stats([x, n, xxT, n])
        else:
            stats = list(map(self.weighted_statistics, data, weights))
            return reduce(add, stats)

    def log_partition(self):
        return 0.5 * np.einsum('d,dl,l->', self.mu, self.lmbda, self.mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x, copy=False).reshape((-1, self.dim))

            log_lik = np.einsum('d,dl,nl->n', self.mu, self.lmbda, x, optimize=True)\
                      - 0.5 * np.einsum('nd,dl,nl->n', x, self.lmbda, x, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition() + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x))

    def expected_statistics(self):
        E_x = self.mu
        E_xxT = np.outer(self.mu, self.mu) + self.sigma
        return E_x, E_xxT

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        x, n, xxT, n = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        self.mu = x / n
        sigma = xxT / n - np.outer(self.mu, self.mu)

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.lmbda = np.linalg.inv(sigma)


class StackedGaussiansWithPrecision:

    def __init__(self, size, dim, mus=None, lmbdas=None):
        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        lmbdas = [None] * self.size if lmbdas is None else lmbdas
        self.dists = [GaussianWithPrecision(dim, mus[k], lmbdas[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.lmbdas

    @params.setter
    def params(self, values):
        self.mus, self.lmbdas = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.mu = value[k, ...]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @lmbdas.setter
    def lmbdas(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda = value[k, ...]

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def mean(self):
        return np.array([dist.mean() for dist in self.dists])

    def mode(self):
        return np.array([dist.mode() for dist in self.dists])

    def rvs(self, sizes):
        return np.vstack([dist.rvs(size) for dist, size in zip(self.dists, sizes)])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, data, fold=True):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            if fold:
                c0, c1 = 'nd->d', 'nd,nl->dl'
                n = data.shape[0]
            else:
                c0, c1 = 'nd->nd', 'nd,nl->ndl'
                n = np.ones((data.shape[0], ))

            x = np.einsum(c0, data, optimize=True)
            xxT = np.einsum(c1, data, data, optimize=True)

            xk = np.array([x for _ in range(self.size)])
            xxTk = np.array([xxT for _ in range(self.size)])
            nk = np.array([n for _ in range(self.size)])

            return Stats([xk, nk, xxTk, nk])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, data))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[:, idx]

            c0, c1 = 'kn,nd->kd', 'nd,kn,nl->kdl'

            xk = np.einsum(c0, weights, data, optimize=True)
            xxTk = np.einsum(c1, data, weights, data, optimize=True)
            nk = np.sum(weights, axis=1)

            return Stats([xk, nk, xxTk, nk])
        else:
            stats = list(map(self.weighted_statistics, data, weights))
            return reduce(add, stats)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x, copy=False).reshape((-1, self.dim))

            log_lik = np.einsum('kd,kdl,nl->kn', self.mus, self.lmbdas, x, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->kn', x, self.lmbdas, x, optimize=True)

            log_lik[:, bads] = 0.
            log_lik += - np.expand_dims(self.log_partition(), axis=1)\
                       + np.expand_dims(self.log_base(), axis=1)
            return log_lik
        else:
            return list(map(self.log_likelihood, x))

    def max_likelihood(self, data, weights):
        xk, nk, xxTk, nk = self.weighted_statistics(data, weights)

        mus = np.zeros((self.size, self.dim))
        lmbdas = np.zeros((self.size, self.dim, self.dim))
        for k in range(self.size):
            mus[k] = xk[k] / nk[k]
            sigma = xxTk[k] / nk[k] - np.outer(mus[k], mus[k])

            # numerical stabilization
            sigma = symmetrize(sigma) + 1e-16 * np.eye(self.dim)
            assert np.allclose(sigma, sigma.T)
            assert np.all(np.linalg.eigvalsh(sigma) > 0.)

            lmbdas[k] = np.linalg.inv(sigma)

        self.mus = mus
        self.lmbdas = lmbdas


class TiedGaussiansWithPrecision(StackedGaussiansWithPrecision, ABC):

    def __init__(self, size, dim, mus=None, lmbdas=None):
        super(TiedGaussiansWithPrecision, self).__init__(size, dim, mus, lmbdas)

    def max_likelihood(self, data, weights):
        xk, nk, xxTk, nk = self.weighted_statistics(data, weights)

        xxT = np.sum(xxTk, axis=0)
        n = np.sum(nk, axis=0)

        mus = np.zeros((self.size, self.dim))
        sigma = np.zeros((self.dim, self.dim))

        sigma += xxT
        for k in range(self.size):
            mus[k] = xk[k] / nk[k]
            sigma -= nk[k] * np.outer(mus[k], mus[k])
        sigma /= n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.mus = mus
        lmbda = np.linalg.inv(sigma)
        self.lmbdas = np.array(self.size * [lmbda])


class GaussianWithDiagonalPrecision(_GaussianBase):

    def __init__(self, dim, mu=None, lmbda_diag=None):

        self._lmbda_diag = lmbda_diag
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

        super(GaussianWithDiagonalPrecision, self).__init__(dim, mu)

    @property
    def params(self):
        return self.mu, self.lmbda_diag

    @params.setter
    def params(self, values):
        self.mu, self.lmbda_diag = values

    @property
    def nb_params(self):
        return self.dim + self.dim

    @staticmethod
    def std_to_nat(params):
        a = params[1] * params[0]
        b = - 0.5 * params[1]
        return Stats([a, b])

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * (1. / natparam[1]) * natparam[0]
        lmbdas = - 2. * natparam[1]
        return mu, lmbdas

    @property
    def lmbda_diag(self):
        return self._lmbda_diag

    @lmbda_diag.setter
    def lmbda_diag(self, value):
        self._lmbda_diag = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def lmbda(self):
        assert self.lmbda_diag is not None
        return np.diag(self.lmbda_diag)

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = np.diag(np.sqrt(self.lmbda_diag))
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = np.diag(1. / np.sqrt(self.lmbda_diag))
        return self._lmbda_chol_inv

    @property
    def sigma_diag(self):
        return 1. / self.lmbda_diag

    @property
    def sigma(self):
        return np.diag(self.sigma_diag)

    def rvs(self, size=1):
        size = self.dim if size == 1 else tuple([size, self.dim])
        return self.mu + npr.normal(size=size).dot(self.lmbda_chol_inv.T)

    def statistics(self, data, fold=True):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            x = np.sum(data, axis=0)
            n = data.shape[0]
            xx = np.einsum('nd,nd->d', data, data)
            nd = np.broadcast_to(data.shape[0], (self.dim, ))

            return Stats([x, nd, nd, xx])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, data))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[idx]

            x = np.einsum('n,nd->d', weights, data)
            n = np.sum(weights)
            xx = np.einsum('nd,n,nd->d', data, weights, data)
            nd = np.broadcast_to(np.sum(weights), (self.dim, ))

            return Stats([x, n, n, xx])
        else:
            stats = list(map(self.weighted_statistics, data, weights))
            return reduce(add, stats)

    def log_partition(self):
        return 0.5 * np.einsum('d,dl,l->', self.mu, self.lmbda, self.mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def expected_statistics(self):
        E_x = self.mu
        E_xx = self.mu**2 + self.sigma_diag
        return E_x, E_xx

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        x, nd, nd, xx = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)

        self.mu = x / nd
        self.lmbda_diag = 1. / (xx / nd - self.mu**2)


class StackedGaussiansWithDiagonalPrecision:

    def __init__(self, size, dim, mus=None, lmbdas_diags=None):
        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        lmbdas_diags = [None] * self.size if lmbdas_diags is None else lmbdas_diags
        self.dists = [GaussianWithDiagonalPrecision(dim, mus[k], lmbdas_diags[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.lmbdas_diags

    @params.setter
    def params(self, values):
        self.mus, self.lmbdas_diags = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.mu = value[k, ...]

    @property
    def lmbdas_diags(self):
        return np.array([dist.lmbda_diag for dist in self.dists])

    @lmbdas_diags.setter
    def lmbdas_diags(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda_diag = value[k, ...]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas_diags(self):
        return np.array([dist.sigma_diag for dist in self.dists])

    @property
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def mean(self):
        return np.array([dist.mean() for dist in self.dists])

    def mode(self):
        return np.array([dist.mode() for dist in self.dists])

    def rvs(self, sizes):
        return np.vstack([dist.rvs(size) for dist, size in zip(self.dists, sizes)])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, data, fold=True):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            if fold:
                c0, c1 = 'nd->d', 'nd,nd->d'
                nd = np.broadcast_to(data.shape[0], (self.dim,))
            else:
                c0, c1 = 'nd->nd', 'nd,nd->nd'
                nd = np.ones((data.shape[0], self.dim))

            x = np.einsum(c0, data, optimize=True)
            xx = np.einsum(c1, data, data, optimize=True)

            xk = np.array([x for _ in range(self.size)])
            xxk = np.array([xx for _ in range(self.size)])
            ndk = np.array([nd for _ in range(self.size)])

            return Stats([xk, ndk, ndk, xxk])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, data))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data, weights = data[idx], weights[:, idx]

            xk = np.einsum('kn,nd->kd', weights, data)
            xxk = np.einsum('nd,kn,nd->kd', data, weights, data)
            ndk = np.broadcast_to(np.sum(weights, axis=1)[:, None], (self.size, self.dim))

            return Stats([xk, ndk, ndk, xxk])
        else:
            stats = list(map(self.weighted_statistics, data, weights))
            return reduce(add, stats)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        if isinstance(x, np.ndarray):
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x, copy=False).reshape((-1, self.dim))

            log_lik = np.einsum('kd,kdl,nl->kn', self.mus, self.lmbdas, x, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->kn', x, self.lmbdas, x, optimize=True)

            log_lik[:, bads] = 0.
            log_lik += - np.expand_dims(self.log_partition(), axis=1)\
                       + np.expand_dims(self.log_base(), axis=1)
            return log_lik
        else:
            return list(map(self.log_likelihood, x))

    def max_likelihood(self, data, weights):
        xk, ndk, ndk, xxk = self.weighted_statistics(data, weights)

        mus = np.zeros((self.size, self.dim))
        lmbdas_diags = np.zeros((self.size, self.dim))
        for k in range(self.size):
            mus[k] = xk[k] / ndk[k]
            lmbdas_diags[k] = 1. / (xxk[k] / ndk[k] - mus[k]**2 + 1e-16)

        self.mus = mus
        self.lmbdas_diags = lmbdas_diags


class TiedGaussiansWithDiagonalPrecision(StackedGaussiansWithDiagonalPrecision, ABC):

    def __init__(self, size, dim, mus=None, lmbdas_diags=None):
        super(TiedGaussiansWithDiagonalPrecision, self).__init__(size, dim, mus, lmbdas_diags)

    def max_likelihood(self, data, weights):
        xk, ndk, ndk, xxk = self.weighted_statistics(data, weights)

        xx = np.sum(xxk, axis=0)
        nd = np.sum(ndk, axis=0)

        mus = np.zeros((self.size, self.dim))
        sigma_diag = np.zeros((self.dim, ))

        sigma_diag += xx
        for k in range(self.size):
            mus[k] = xk[k] / ndk[k]
            sigma_diag -= ndk[k] * mus[k]**2
        sigma_diag /= nd

        self.mus = mus
        lmbda_diag = 1. / (sigma_diag + 1e-16)
        self.lmbdas_diags = np.array(self.size * [lmbda_diag])
