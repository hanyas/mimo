import numpy as np
import numpy.random as npr

from scipy import linalg
from numpy.core.umath_tests import inner1d

from mimo.abstractions import Distribution
from mimo.util.general import near_pd


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
        return self.M, self.U, self.V

    @params.setter
    def params(self, values):
        self.M, self.U, self.V = values

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
            self._U_chol = np.linalg.cholesky(near_pd(self.U))
        return self._U_chol

    @property
    def V_chol(self):
        if self._V_chol is None:
            self._V_chol = np.linalg.cholesky(near_pd(self.V))
        return self._V_chol

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
        return self._sigma_chol

    def rvs(self, size=None):
        if size is None:
            aux = npr.normal(size=self.drow * self.dcol).dot(self.sigma_chol.T)
            return self.M + np.reshape(aux, (self.drow, self.dcol), order='F')
        else:
            size = tuple([size, self.drow * self.dcol])
            aux = npr.normal(size=self.size).dot(self.sigma_chol.T)
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

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            xxT = np.einsum('nk,nh->kh', data, data)
            x = data.sum(0)
            n = data.shape[0]
            return np.array([x, xxT, n])
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            xxT = np.einsum('nk,n,nh->kh', data, weights, data)
            x = weights.dot(data)
            n = weights.sum()
            return np.array([x, xxT, n])
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.dim, )),
                         np.zeros((self.dim, self.dim)), 0])

    def log_partition(self):
        return 0.5 * self.drow * self.dcol * np.log(2. * np.pi)\
               + self.drow * np.sum(np.log(np.diag(self.V_chol)))\
               + self.dcol * np.sum(np.log(np.diag(self.U_chol)))

    def entropy(self):
        return NotImplementedError
