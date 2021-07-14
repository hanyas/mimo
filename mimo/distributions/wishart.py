import numpy as np
import numpy.random as npr

import scipy as sc
from scipy.special import multigammaln, digamma
from scipy.linalg.lapack import get_lapack_funcs

from mimo.utils.abstraction import Statistics as Stats


class Wishart:

    def __init__(self, dim, psi=None, nu=None):
        self.dim = dim

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
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        a = - 0.5 * np.linalg.inv(params[0])
        b = 0.5 * (params[1] - a.shape[0] - 1)
        return Stats([a, b])

    @staticmethod
    def nat_to_std(natparam):
        psi = - 0.5 * np.linalg.inv(natparam[0])
        nu = 2. * natparam[1] + psi.shape[0] + 1
        return psi, nu

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

    def mean(self):
        return self.nu * self.psi

    def mode(self):
        assert self.nu >= (self.dim + 1)
        return (self.nu - self.dim - 1) * self.psi

    # copied from scipy
    def rvs(self, size=1):
        # Random normal variates for off-diagonal elements
        n_tril = self.dim * (self.dim - 1) // 2
        covariances = npr.normal(size=n_tril).reshape((n_tril,))

        # Random chi-square variates for diagonal elements
        variances = (np.r_[[npr.chisquare(self.nu - (i + 1) + 1, size=1)**0.5
                            for i in range(self.dim)]].reshape((self.dim,)).T)

        A = np.zeros((self.dim, self.dim))

        # Input the covariances
        tril_idx = np.tril_indices(self.dim, k=-1)
        A[tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(self.dim)
        A[diag_idx] = variances

        T = np.dot(self.psi_chol, A)
        return np.dot(T, T.T)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            x = data
            logdet_x = np.linalg.slogdet(data)[1]
            n = np.ones((data.shape[0], ))

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
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

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               + self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def log_likelihood(self, x):
        log_lik = 0.5 * (self.nu - self.dim - 1) * np.linalg.slogdet(x)[1]\
                  - 0.5 * np.trace(np.linalg.solve(self.psi, x))
        return - self.log_partition() + self.log_base() + log_lik

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


class InverseWishart:

    def __init__(self, dim, psi=None, nu=None):
        self.dim = dim

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
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        a = - 0.5 * params[0]
        b = - 0.5 * (params[1] + a.shape[0] + 1)
        return Stats([a, b])

    @staticmethod
    def nat_to_std(natparam):
        psi = - 2. * natparam[0]
        nu = - (2. * natparam[1] + psi.shape[0] + 1)
        return psi, nu

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

    def mean(self):
        assert self.nu > (self.dim + 1)
        return self.psi / (self.nu - self.dim - 1.)

    def mode(self):
        return self.psi / (self.nu + self.dim + 1.)

    # copied from scipy
    def rvs(self, size=1):
        # Random normal variates for off-diagonal elements
        n_tril = self.dim * (self.dim - 1) // 2
        covariances = npr.normal(size=n_tril).reshape((n_tril,))

        # Random chi-square variates for diagonal elements
        variances = (np.r_[[npr.chisquare(self.nu - (i + 1) + 1, size=1)**0.5
                            for i in range(self.dim)]].reshape((self.dim,)).T)

        A = np.zeros((self.dim, self.dim))

        # Input the covariances
        tril_idx = np.tril_indices(self.dim, k=-1)
        A[tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(self.dim)
        A[diag_idx] = variances

        eye = np.eye(self.dim)

        L, lower = sc.linalg.cho_factor(self.psi, lower=True)
        inv_scale = sc.linalg.cho_solve((L, lower), eye)
        C = sc.linalg.cholesky(inv_scale, lower=True)

        trtrs = get_lapack_funcs(('trtrs'), (A,))

        T = np.dot(C, A)
        if self.dim > 1:
            T, _ = trtrs(T, eye, lower=True)
        else:
            T = 1. / T

        return np.dot(T.T, T)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            x = np.sum(np.linalg.inv(data), axis=0)
            logdet_x = np.linalg.slogdet(data)[1]
            n = np.ones((data.shape[0], ))

            return Stats([x, n, logdet_x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
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

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               - self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def log_likelihood(self, x):
        log_lik = - 0.5 * (self.nu + self.dim + 1) * np.linalg.slogdet(x)[1]\
                  - 0.5 * np.trace(self.psi @ np.linalg.inv(x))
        return - self.log_partition() + self.log_base() + log_lik

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
        return dist.log_partition() - dist.log_base() \
               - (np.tensordot(nat_param[0], stats[0]) + nat_param[1] * stats[1])
