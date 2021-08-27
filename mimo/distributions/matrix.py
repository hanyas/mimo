import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg

from mimo.utils.abstraction import Statistics as Stats


class MatrixNormalWithPrecision:

    def __init__(self, column_dim, row_dim,
                 M=None, V=None, K=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.M = M
        self._V = V
        self._K = K

        self._V_chol = None
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.M, self.V, self.K

    @params.setter
    def params(self, values):
        self.M, self.V, self.K = values

    @property
    def nb_params(self):
        num = self.column_dim * self.row_dim
        return num + num * (num + 1) / 2

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        a = params[1] @ params[0]
        b = - 0.5 * params[1]
        return a, b

    @staticmethod
    def nat_to_std(natparam):
        mu = - 0.5 * np.linalg.inv(natparam[1]) @ natparam[0]
        lmbda = - 2. * natparam[1]
        return Stats([mu, lmbda])

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value
        self._V_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = sc.linalg.cholesky(self.V, lower=False)
        return self._V_chol

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)
        return self._K_chol

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def rvs(self):
        aux = npr.normal(size=self.row_dim * self.column_dim).dot(self.lmbda_chol_inv.T)
        return self.M + np.reshape(aux, (self.row_dim, self.column_dim), order='F')

    @property
    def base(self):
        return np.power(2. * np.pi, - self.row_dim * self.column_dim / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.row_dim * self.column_dim), order='F')
        return 0.5 * np.einsum('d,dl,l->', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.row_dim * self.column_dim), order='F')
        mu = np.reshape(self.M, (self.row_dim * self.column_dim), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xr = np.nan_to_num(xr, copy=False).reshape((-1, self.row_dim * self.column_dim))

        log_lik = np.einsum('d,dl,nl->n', mu, self.lmbda, xr)\
                  - 0.5 * np.einsum('nd,dl,nl->n', xr, self.lmbda, xr)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    def expected_statistics(self):
        mu = np.reshape(self.M, (self.row_dim * self.column_dim), order='F')
        E_x = mu
        E_xxT = np.outer(mu, mu) + self.sigma
        return E_x, E_xxT

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.tensordot(nat_param[1], stats[1]))

    def relative_entropy(self, dist):
        kl = 0.5 * np.trace(dist.K @ np.linalg.inv(self.K))\
             - self.column_dim * self.row_dim
        kl += self.row_dim * np.sum(np.log(np.diag(dist.K_chol)))
        kl -= self.row_dim * np.sum(np.log(np.diag(self.K_chol)))
        return kl


class MatrixNormalWithDiagonalPrecision:

    def __init__(self, column_dim, row_dim,
                 M=None, V_diag=None, K=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.M = M

        self._V_diag = V_diag
        self._K = K

        self._V_chol = None
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.M, self.V_diag, self.K

    @params.setter
    def params(self, values):
        self.M, self.V_diag, self.K = values

    @property
    def nb_params(self):
        return self.column_dim * self.row_dim\
               + self.column_dim * self.row_dim

    @property
    def V_diag(self):
        return self._V_diag

    @V_diag.setter
    def V_diag(self, value):
        self._V_diag = value
        self._V_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def V(self):
        return np.diag(self.V_diag)

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = sc.linalg.cholesky(self.V, lower=False)
        return self._V_chol

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def K_chol(self):
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)
        return self._K_chol

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def rvs(self):
        aux = npr.normal(size=self.row_dim * self.column_dim).dot(self.lmbda_chol_inv.T)
        return self.M + np.reshape(aux, (self.row_dim, self.column_dim), order='F')

    @property
    def base(self):
        return np.power(2. * np.pi, - self.row_dim * self.column_dim / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.row_dim * self.column_dim), order='F')
        return 0.5 * np.einsum('d,dl,l->', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (self.row_dim * self.column_dim, ), order='F')
        mu = np.reshape(self.M, (self.row_dim * self.column_dim), order='F')

        log_lik = np.einsum('d,dl,l->', mu, self.lmbda, xr)\
                 - 0.5 * np.einsum('d,dl,l->', xr, self.lmbda, xr)

        return - self.log_partition() + self.log_base() + log_lik
