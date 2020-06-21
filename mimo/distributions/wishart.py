import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats
from scipy.special import multigammaln, digamma

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats


class Wishart(Distribution):

    def __init__(self, psi, nu):
        self.nu = nu

        self._psi = psi
        self._psi_chol = None

    @property
    def params(self):
        return self.psi, self.nu

    @params.setter
    def params(self, values):
        self.psi, self.nu = values

    @property
    def dim(self):
        return self.psi.shape[0]

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value
        self._psi_chol = None

    @property
    def psi_chol(self):
        if self._psi_chol is None:
            self._psi_chol = np.linalg.cholesky(self.psi)

        return self._psi_chol

    def rvs(self, size=1):
        assert size == 1

        # # This is slow
        # return stats.wishart(df=self.nu, scale=self.psi).rvs(size).reshape(self.dim, self.dim)

        # This is faster
        # use matlab's heuristic for choosing between the two different sampling schemes
        if (self.nu <= 81 + self.dim) and (self.nu == np.round(self.nu)):
            # direct
            X = np.dot(self.psi_chol, np.random.normal(size=(int(self.dim), int(self.nu))))
        else:
            A = np.diag(np.sqrt(npr.chisquare(self.nu - np.arange(self.dim))))
            A[np.tri(self.dim, k=-1, dtype=bool)] = npr.normal(size=int(self.dim * (self.dim - 1) / 2.))
            X = np.dot(self.psi_chol, A)

        return np.dot(X, X.T)

    def mean(self):
        return self.nu * self.psi

    def mode(self):
        assert self.nu >= (self.dim + 1)
        return (self.nu - self.dim - 1) * self.psi

    def log_likelihood(self, x):
        # not vectorized
        log_lik = 0.5 * (self.nu - self.dim - 1) * np.linalg.slogdet(x)[1]\
                  - 0.5 * np.trace(sc.linalg.solve(self.psi, x))
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            x = data
            logdet_x = np.linalg.slogdet(data)[1]
            n = np.ones((data.shape[0], ))

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            x = np.einsum('n,nkh->nkh', weights, data)
            logdet_x = np.einsum('n,nkh->nkh', weights,
                                 np.linalg.slogdet(data)[1])
            n = weights

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.weighted_statistics, data, weights))

    @property
    def base(self):
        return 1.

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
        psi = - 0.5 * np.linalg.inv(params[0])
        nu = 0.5 * (params[1] - psi.shape[0] - 1)
        return Stats([psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        psi = - 0.5 * np.linalg.inv(natparam[0])
        nu = 2. * natparam[1] + psi.shape[0] + 1
        return psi, nu

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               + self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def expected_statistics(self):
        E_X = self.nu * self.psi
        E_logdet_X = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                     + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.psi_chol)))
        return E_X, E_logdet_X

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.tensordot(nat_param[0], stats[0]) + nat_param[1] * stats[1])

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base()\
               - (np.tensordot(nat_param[0], stats[0]) + nat_param[1] * stats[1])


class InverseWishart(Distribution):

    def __init__(self, psi, nu):
        self.nu = nu

        self._psi = psi
        self._psi_chol = None

    @property
    def params(self):
        return self.psi, self.nu

    @params.setter
    def params(self, values):
        self.psi, self.nu = values

    @property
    def dim(self):
        return self.psi.shape[0]

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value
        self._psi_chol = None

    @property
    def psi_chol(self):
        if self._psi_chol is None:
            self._psi_chol = np.linalg.cholesky(self.psi)
        return self._psi_chol

    def rvs(self, size=1):
        assert size == 1

        # # This is slow
        # return stats.invwishart(df=self.nu, scale=self.psi).rvs(size).reshape(self.dim, self.dim)

        # This is faster
        if (self.nu <= 81 + self.dim) and (self.nu == np.round(self.nu)):
            x = npr.randn(int(self.nu), self.dim)
        else:
            x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(self.nu - np.arange(self.dim)))))
            x[np.triu_indices_from(x, 1)] = npr.randn(self.dim * (self.dim - 1) // 2)
        R = np.linalg.qr(x, 'r')
        T = sc.linalg.solve_triangular(R.T, self.psi_chol.T, lower=True).T

        return np.dot(T, T.T)

    def mean(self):
        assert self.nu > (self.dim + 1)
        return self.psi / (self.nu - self.dim - 1.)

    def mode(self):
        return self.psi / (self.nu + self.dim + 1.)

    def log_likelihood(self, x):
        # not vectorized
        log_lik = - 0.5 * (self.nu + self.dim + 1) * np.linalg.slogdet(x)[1]\
                  - 0.5 * np.trace(self.psi @ np.linalg.inv(x))
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            x = np.sum(np.linalg.inv(data), axis=0)
            logdet_x = np.linalg.slogdet(data)[1]
            n = np.ones((data.shape[0], ))

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            x = np.einsum('n,nkh->nkh', weights, np.linalg.inv(data))
            logdet_x = np.einsum('n,nkh->nkh', weights,
                                 np.linalg.slogdet(data)[1])
            n = weights

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.weighted_statistics, data, weights))

    @property
    def base(self):
        return 1.

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
        psi = - 0.5 * params[0]
        nu = - 0.5 * (params[1] + psi.shape[0] + 1)
        return Stats([psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        psi = - 2. * natparam[0]
        nu = - (2. * natparam[1] + psi.shape[0] + 1)
        return psi, nu

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               - self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def expected_statistics(self):
        E_X = self.psi / (self.nu - self.dim - 1)
        E_logdet_X = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                       + self.dim * np.log(2.) - 2. * np.sum(np.log(np.diag(self.psi_chol)))
        return E_X, E_logdet_X

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.tensordot(nat_param[0], stats[0]) + nat_param[1] * stats[1])

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base()\
               - (np.tensordot(nat_param[0], stats[0]) + nat_param[1] * stats[1])
