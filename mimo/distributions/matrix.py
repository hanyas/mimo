import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import linalg

from mimo.abstraction import Distribution


class MatrixNormalWithPrecision(Distribution):

    def __init__(self, M=None, V=None, K=None):
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
        num = self.dcol * self.drow
        return num + num * (num + 1) / 2

    @property
    def dcol(self):
        return self.M.shape[1]

    @property
    def drow(self):
        return self.M.shape[0]

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
        # upper cholesky triangle
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
        # upper cholesky triangle
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)
        return self._K_chol

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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
        return self._lmbda_chol_inv

    def rvs(self, size=1):
        if size == 1:
            aux = npr.normal(size=self.drow * self.dcol).dot(self.lmbda_chol_inv.T)
            return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')
        else:
            size = tuple([size, self.drow * self.dcol])
            aux = npr.normal(size=size).dot(self.lmbda_chol_inv.T)
            return self.M + np.reshape(aux, (size, self.drow, self.dcol), order='F')

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.drow * self.dcol), order='F')
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xr = np.nan_to_num(xr, copy=False).reshape((-1, self.drow * self.dcol))

        log_lik = np.einsum('k,kh,nh->n', mu, self.lmbda, xr)\
                  - 0.5 * np.einsum('nk,kh,nh->n', xr, self.lmbda, xr)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow * self.dcol / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')
        return 0.5 * np.einsum('k,kh,h->', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def entropy(self):
        raise NotImplementedError


class MatrixNormalWithDiagonalPrecision(Distribution):
    def __init__(self, M=None, vs=None, K=None):
        self.M = M

        self._vs = vs
        self._K = K

        self._V_chol = None
        self._K_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.M, self.vs, self.K

    @params.setter
    def params(self, values):
        self.M, self.vs, self.K = values

    @property
    def nb_params(self):
        return self.dcol * self.drow\
               + self.dcol * self.drow

    @property
    def dcol(self):
        return self.M.shape[1]

    @property
    def drow(self):
        return self.M.shape[0]

    @property
    def vs(self):
        return self._vs

    @vs.setter
    def vs(self, value):
        self._vs = value
        self._V_chol = None

        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def V(self):
        assert self._vs is not None
        return np.diag(self._vss)

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = np.diag(np.sqrt(self._vs))
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
        # upper cholesky triangle
        if self._K_chol is None:
            self._K_chol = sc.linalg.cholesky(self.K, lower=False)

    @property
    def lmbda(self):
        return np.kron(self.K, self.V)

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
        return self._lmbda_chol_inv

    def rvs(self, size=1):
        if size == 1:
            aux = npr.normal(size=self.drow * self.dcol).dot(self.lmbda_chol_inv.T)
            return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')
        else:
            size = tuple([size, self.drow * self.dcol])
            aux = npr.normal(size=size).dot(self.lmbda_chol_inv.T)
            return self.M + np.reshape(aux, (size, self.drow, self.dcol), order='F')

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.drow * self.dcol), order='F')
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xr = np.nan_to_num(xr, copy=False).reshape((-1, self.drow * self.dcol))

        log_lik = np.einsum('k,kh,nh->n', mu, self.omega, xr)\
                  - 0.5 * np.einsum('nk,kh,nh->n', xr, self.omega, xr)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow * self.dcol / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        mu = np.reshape(self.M, (self.drow * self.dcol), order='F')
        return 0.5 * np.einsum('k,kh,h->', mu, self.omega, mu)\
               - np.sum(np.log(np.diag(self.omega_chol)))

    def entropy(self):
        raise NotImplementedError
