import numpy as np
import scipy as sc
import numpy.random as npr

from scipy import stats
from scipy.special import multigammaln, digamma

from mimo.abstractions import Distribution


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

    def rvs(self, size=None):
        # use matlab's heuristic for choosing between the two different sampling schemes
        if (self.nu <= 81 + self.dim) and (self.nu == round(self.nu)):
            # direct
            X = np.dot(self.psi_chol, np.random.normal(size=(self.dim, self.nu)))
        else:
            A = np.diag(np.sqrt(npr.chisquare(self.nu - np.arange(self.dim))))
            A[np.tri(self.dim, k=-1, dtype=bool)] =\
                npr.normal(size=(self.dim * (self.dim - 1) / 2.))
            X = np.dot(self.psi_chol, A)

        return np.dot(X, X.T)

    def mean(self):
        return self.nu * self.psi

    def mode(self):
        return (self.nu - self.dim - 1) * self.psi

    def log_likelihood(self, x):
        x_det = np.linalg.det(x)

        loglik = - 0.5 * self.nu * self.dim * np.log(2.)\
                 - 0.5 * self.nu * np.sum(np.log(np.diag(self.psi_chol)))\
                 - multigammaln(self.nu / 2., self.dim)\
                 + 0.5 * (self.nu - self.dim - 1) * np.log(x_det)\
                 - 0.5 * np.trace(sc.linalg.solve(self.psi, x))

        return loglik

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               + self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def entropy(self):
        Elogdetlmbda = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                       + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.psi_chol)))
        aux = - 0.5 * (self.nu - self.dim - 1) * Elogdetlmbda + 0.5 * self.nu * self.dim
        return self.log_partition() + aux


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

    def rvs(self, size=None):
        if (self.nu <= 81 + self.dim) and (self.nu == np.round(self.nu)):
            x = npr.randn(int(self.nu), self.dim)
        else:
            x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(self.nu - np.arange(self.dim)))))
            x[np.triu_indices_from(x, 1)] = npr.randn(self.dim * (self.dim - 1) // 2)
        R = np.linalg.qr(x, 'r')
        T = sc.linalg.solve_triangular(R.T, self.psi_chol.T, lower=True).T
        return np.dot(T, T.T)

    def mean(self):
        return self.psi / (self.nu - self.dim - 1.)

    def mode(self):
        return self.psi / (self.nu + self.dim + 1.)

    def log_likelihood(self, x):
        x_det = np.linalg.det(x)

        loglik = - 0.5 * self.nu * self.dim * np.log(2.)\
                 + 0.5 * self.nu * np.sum(np.log(np.diag(self.psi_chol)))\
                 - multigammaln(self.nu / 2., self.dim)\
                 - 0.5 * (self.nu + self.dim + 1) * np.log(x_det)\
                 - 0.5 * np.trace(self.psi @ np.linalg.inv(x))

        return loglik

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               - self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def entropy(self):
        Elogdetlmbda = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                       + self.dim * np.log(2.) - 2. * np.sum(np.log(np.diag(self.psi_chol)))
        aux = - 0.5 * (self.nu - self.dim - 1) * Elogdetlmbda + 0.5 * self.nu * self.dim
        return self.log_partition() + aux
