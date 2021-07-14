import numpy as np
import numpy.random as npr

import scipy as sc

from operator import add
from functools import reduce, partial

from mimo.utils.abstraction import Statistics as Stats
from mimo.utils.matrix import symmetrize


class LinearGaussianWithPrecision:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and precision matrix:
        A, lmbda
    """

    def __init__(self, A=None, lmbda=None, affine=True):

        self.A = A
        self.affine = affine

        self._lmbda = lmbda
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.A, self.lmbda

    @params.setter
    def params(self, values):
        self.A, self.lmbda = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.A.shape[1] - 1
        else:
            return self.A.shape[1]

    @property
    def drow(self):
        # output dimension
        return self.A.shape[0]

    @property
    def nb_params(self):
        return self.dcol * self.drow \
               + self.drow * (self.drow + 1) / 2

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
        # upper cholesky triangle
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

    def rvs(self, x=None, size=None):
        assert x is not None
        size = 1 if x.ndim == 1 else x.shape[0]

        y = self.mean(x)
        y += npr.normal(size=(size, self.drow)).dot(self.lmbda_chol_inv.T)

        return y

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('kh,...h->...k', A, x, optimize=True) + b.T
        else:
            y = np.einsum('kh,...h->...k', self.A, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def log_likelihood(self, y, x):
        assert x is not None

        bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                              np.isnan(np.atleast_2d(y)).any(axis=1))

        x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
        y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

        mu = self.mean(x)
        log_lik = np.einsum('nk,kh,nh->n', mu, self.lmbda, y, optimize=True)\
                  - 0.5 * np.einsum('nk,kh,nh->n', y, self.lmbda, y, optimize=True)

        log_lik[bads] = 0
        return - self.log_partition(x) + self.log_base() + log_lik

    def statistics(self, y, x, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x = y[idx], x[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if vectorize:
                contract = 'nk,nh->nkh'
                n = np.ones((y.shape[0], ))
            else:
                contract = 'nk,nh->kh'
                n = y.shape[0]

            yxT = np.einsum(contract, y, x, optimize=True)
            xxT = np.einsum(contract, x, x, optimize=True)
            yyT = np.einsum(contract, y, y, optimize=True)

            return Stats([yxT, xxT, yyT, n])
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, y, x))
            return stats if vectorize else reduce(add, stats)

    def weighted_statistics(self, y, x, weights, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x, weights = y[idx], x[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if vectorize:
                contract = 'nk,n,nh->nkh'
                n = weights
            else:
                contract = 'nk,n,nh->kh'
                n = np.sum(weights)

            yxT = np.einsum(contract, y, weights, x, optimize=True)
            xxT = np.einsum(contract, x, weights, x, optimize=True)
            yyT = np.einsum(contract, y, weights, y, optimize=True)

            return Stats([yxT, xxT, yyT, n])
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = list(map(func, y, x, weights))
            return stats if vectorize else reduce(add, stats)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nk,kh,nh->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def entropy(self, x):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, y, x, weights=None):
        stats = self.statistics(y, x) if weights is None\
            else self.weighted_statistics(y, x, weights)

        yxT, xxT, yyT, n = stats
        self.A = np.linalg.solve(xxT, yxT.T).T
        sigma = (yyT - self.A.dot(yxT.T)) / n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.drow)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.lmbda = np.linalg.inv(sigma)

        return self


class LinearGaussianWithDiagonalPrecision:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and diagonal precision matrix:
        A, lmbdas
    """

    def __init__(self, A=None, lmbdas=None, affine=True):

        self.A = A
        self.affine = affine

        self._lmbdas = lmbdas
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.A, self.lmbdas

    @params.setter
    def params(self, values):
        self.A, self.lmbdas = values

    @property
    def dcol(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.A.shape[1] - 1
        else:
            return self.A.shape[1]

    @property
    def drow(self):
        # output dimension
        return self.A.shape[0]

    @property
    def nb_params(self):
        return self.dcol * self.drow + self.drow

    @property
    def lmbdas(self):
        return self._lmbdas

    @lmbdas.setter
    def lmbdas(self, value):
        self._lmbdas = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

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

    def rvs(self, x=None, size=None):
        assert x is not None
        size = 1 if x.ndim == 1 else x.shape[0]

        y = self.mean(x)
        y += npr.normal(size=(size, self.drow)).dot(self.lmbda_chol_inv.T)

        return y

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('kh,...h->...k', A, x, optimize=True) + b.T
        else:
            y = np.einsum('kh,...h->...k', self.A, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def log_likelihood(self, y, x):
        assert x is not None

        bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                              np.isnan(np.atleast_2d(y)).any(axis=1))

        x = np.nan_to_num(x, copy=False).reshape((-1, self.dcol))
        y = np.nan_to_num(y, copy=False).reshape((-1, self.drow))

        mu = self.mean(x)
        log_lik = np.einsum('nk,kh,nh->n', mu, self.lmbda, y, optimize=True)\
                  - 0.5 * np.einsum('nk,kh,nh->n', y, self.lmbda, y, optimize=True)

        log_lik[bads] = 0
        return - self.log_partition(x) + self.log_base() + log_lik

    def statistics(self, y, x, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x = y[idx], x[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if vectorize:
                c0, c1 = 'nk,nh->nkh', 'nk,nk->nk'
                n = np.ones((y.shape[0], ))
            else:
                c0, c1 = 'nk,nh->kh', 'nk,nk->k'
                n = y.shape[0]

            yxT = np.einsum(c0, y, x, optimize=True)
            xxT = np.einsum(c0, x, x, optimize=True)
            yy = np.einsum(c1, y, y, optimize=True)

            return Stats([yxT, xxT, yy, n])
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, y, x))
            return stats if vectorize else reduce(add, stats)

    def weighted_statistics(self, y, x, weights, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x, weights = y[idx], x[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if vectorize:
                c0, c1 = 'nk,n,nh->nkh', 'nk,n,nk->nk'
                n = weights
            else:
                c0, c1 = 'nk,n,nh->kh', 'nk,n,nk->k'
                n = np.sum(weights)

            yxT = np.einsum(c0, y, weights, x, optimize=True)
            xxT = np.einsum(c0, x, weights, x, optimize=True)
            yy = np.einsum(c1, y, weights, y, optimize=True)

            return Stats([yxT, xxT, yy, n])
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = list(map(func, y, x, weights))
            return stats if vectorize else reduce(add, stats)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nk,kh,nh->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def entropy(self, x):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, y, x, weights=None):
        stats = self.statistics(y, x) if weights is None\
            else self.weighted_statistics(y, x, weights)

        yxT, xxT, yy, n = stats
        self.A = np.linalg.solve(xxT, yxT.T).T
        sigmas = (yy - np.einsum('kh,kh->k', self.A, yxT)) / n
        self.lmbdas = 1. / sigmas

        return self


class LinearGaussianWithKnownPrecision(LinearGaussianWithPrecision):
    """
    Multivariate Gaussian distribution with a linear mean function
    and a fixed precision matrix.
    Parameters are linear transf. and precision matrix:
        A, lmbda
    """

    def __init__(self, A=None, lmbda=None, affine=True):
        super(LinearGaussianWithKnownPrecision, self).__init__(A=A, lmbda=lmbda, affine=affine)

    @property
    def params(self):
        return self.A

    @params.setter
    def params(self, values):
        self.A = values

    def statistics(self, y, x, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x = y[idx], x[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxT = np.einsum('nk,nh->kh', y, x, optimize=True)
            xxT = np.einsum('nk,nh->kh', x, x, optimize=True)

            return Stats([yxT, xxT])
        else:
            func = partial(self.statistics, vectorize=vectorize)
            stats = list(map(func, y, x))
            return stats if vectorize else reduce(add, stats)

    def weighted_statistics(self, y, x, weights, vectorize=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x, weights = y[idx], x[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxT = np.einsum('nk,n,nh->kh', y, weights, x, optimize=True)
            xxT = np.einsum('nk,n,nh->kh', x, weights, x, optimize=True)

            return Stats([yxT, xxT])
        else:
            func = partial(self.weighted_statistics, vectorize=vectorize)
            stats = list(map(func, y, x, weights))
            return stats if vectorize else reduce(add, stats)

    def max_likelihood(self, y, x, weights=None):
        raise NotImplementedError
