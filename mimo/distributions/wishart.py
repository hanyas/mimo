import numpy as np
import scipy as sc

from scipy import stats
from scipy.special import multigammaln, digamma

from mimo.abstraction import Distribution
from mimo.util.matrix import near_pd


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
            # self._psi_chol = np.linalg.cholesky(near_pd(self.psi))
            self._psi_chol = np.linalg.cholesky(self.psi)

        return self._psi_chol

    def rvs(self, size=1):
        return stats.wishart(df=self.nu, scale=self.psi).rvs(size).reshape(self.dim, self.dim)

    def mean(self):
        return self.nu * self.psi

    def mode(self):
        assert self.nu >= (self.dim + 1)
        return (self.nu - self.dim - 1) * self.psi

    def log_likelihood(self, x):
        loglik = + 0.5 * (self.nu - self.dim - 1) * np.linalg.slogdet(x)[1]\
                 - 0.5 * np.trace(sc.linalg.solve(self.psi, x))
        return loglik - self.log_partition()

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               + self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def entropy(self):
        E_logdet_psi = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                       + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.psi_chol)))
        entropy = - 0.5 * (self.nu - self.dim - 1) * E_logdet_psi + 0.5 * self.nu * self.dim
        return entropy + self.log_partition()


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
            # self._psi_chol = np.linalg.cholesky(near_pd(self.psi))
            self._psi_chol = np.linalg.cholesky(self.psi)
        return self._psi_chol

    def rvs(self, size=1):
        return stats.invwishart(df=self.nu, scale=self.psi).rvs(size).reshape(self.dim, self.dim)

    def mean(self):
        assert self.nu > (self.dim + 1)
        return self.psi / (self.nu - self.dim - 1.)

    def mode(self):
        return self.psi / (self.nu + self.dim + 1.)

    def log_likelihood(self, x):
        loglik = - 0.5 * (self.nu + self.dim + 1) * np.linalg.slogdet(x)[1]\
                 - 0.5 * np.trace(self.psi @ np.linalg.inv(x))
        return loglik - self.log_partition()

    def log_partition(self):
        return 0.5 * self.nu * self.dim * np.log(2)\
               + multigammaln(self.nu / 2., self.dim)\
               - self.nu * np.sum(np.log(np.diag(self.psi_chol)))

    def entropy(self):
        E_logdet_psi = np.sum(digamma((self.nu - np.arange(self.dim)) / 2.))\
                       + self.dim * np.log(2.) - 2. * np.sum(np.log(np.diag(self.psi_chol)))
        entropy = - 0.5 * (self.nu - self.dim - 1) * E_logdet_psi + 0.5 * self.nu * self.dim
        return entropy + self.log_partition()
