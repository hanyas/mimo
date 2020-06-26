import numpy as np
import numpy.random as npr

from scipy import linalg
from numpy.core.umath_tests import inner1d

from mimo.abstraction import Distribution
from mimo.util.matrix import nearpd


class MatrixNormal(Distribution):

    def __init__(self, M=None, U=None, V=None):
        self.M = M
        self._U = U
        self._V = V

        self._U_chol = None
        self._V_chol = None

        self._sigma_chol = None

    @property
    def params(self):
        return self.M, self.V

    @params.setter
    def params(self, values):
        self.M, self.V = values

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
    def U(self):
        return self._U

    @U.setter
    def U(self, value):
        self._U = value
        # reset Cholesky for new values of U
        # A new Cholesky will be computed when needed
        self._U_chol = None

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value
        # reset Cholesky for new values of V
        # A new Cholesky will be computed when needed
        self._V_chol = None

    @property
    def sigma(self):
        return np.kron(self.V, self.U)

    @property
    def U_chol(self):
        if self._U_chol is None:
            # self._U_chol = np.linalg.cholesky(near_pd(self.U))
            self._U_chol = np.linalg.cholesky(self.U)
        return self._U_chol

    @property
    def V_chol(self):
        if self._V_chol is None:
            # self._V_chol = np.linalg.cholesky(near_pd(self.V))
            self._V_chol = np.linalg.cholesky(self.V)
        return self._V_chol

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            # self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    def rvs(self, size=1):
        if size == 1:
            aux = npr.normal(size=self.drow * self.dcol).dot(self.sigma_chol.T)
            return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')
        else:
            size = tuple([size, self.drow * self.dcol])
            aux = npr.normal(size=size).dot(self.sigma_chol.T)
            return self.M + np.reshape(aux, (size, self.drow, self.dcol), order='F')

    def mean(self):
        return self.M

    def mode(self):
        return self.M

    def log_likelihood(self, x):
        # apply vector operator with Fortran convention
        xr = np.reshape(x, (-1, self.drow * self.dcol), order='F')
        mr = np.reshape(self.M, (self.drow * self.dcol), order='F')

        # Gaussian likelihood on vector dist.
        bads = np.isnan(np.atleast_2d(xr)).any(axis=1)
        xc = np.nan_to_num(xr).reshape((-1, self.dim)) - mr
        xs = linalg.solve_triangular(self.sigma_chol, xc.T, lower=True)
        out = - 0.5 * self.drow * self.dcol * np.log(2. * np.pi)\
              - np.sum(np.log(np.diag(self.sigma_chol))) - 0.5 * inner1d(xs.T, xs.T)
        out[bads] = 0
        return out

    def log_partition(self):
        return 0.5 * self.drow * self.dcol * np.log(2. * np.pi)\
               + self.drow * np.trace(np.log(self.V_chol))\
               + self.dcol * np.trace(np.log(self.U_chol))

    def entropy(self):
        return NotImplementedError
