import numpy as np
import numpy.random as npr

import scipy as sc

from operator import add
from functools import reduce, partial

from mimo.abstraction import Conditional
from mimo.abstraction import Statistics as Stats

from mimo.util.matrix import invpd, symmetrize


class LinearGaussianWithPrecision(Conditional):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
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

    def rvs(self, x=None):
        assert x is not None
        size = 1 if x.ndim == 1 else x.shape[0]

        y = self.mean(x)
        y += npr.normal(size=(size, self.drow)).dot(self.lmbda_chol_inv.T)

        return y

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('kh,...h->...k', A, x) + b.T
        else:
            y = np.einsum('kh,...h->...k', self.A, x)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def log_likelihood(self, y, x):
        assert x is not None

        bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                              np.isnan(np.atleast_2d(y)).any(axis=1))

        x = np.nan_to_num(x).reshape((-1, self.dcol))
        y = np.nan_to_num(y).reshape((-1, self.drow))

        mu = self.mean(x)
        log_lik = np.einsum('nk,kh,nh->n', mu, self.lmbda, y)\
                  - 0.5 * np.einsum('nk,kh,nh->n', mu, self.lmbda, mu)\
                  - 0.5 * np.einsum('nk,kh,nh->n', y, self.lmbda, y)

        log_lik[bads] = 0
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, y, x, keepdim=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x = y[idx], x[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxT = np.einsum('nk,nh->nkh', y, x)
            xxT = np.einsum('nk,nh->nkh', x, x)
            yyT = np.einsum('nk,nh->nkh', y, y)
            n = np.ones((y.shape[0], ))

            if not keepdim:
                yxT = np.sum(yxT, axis=0)
                xxT = np.sum(xxT, axis=0)
                yyT = np.sum(yyT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([yxT, xxT, yyT, n])
        else:
            func = partial(self.statistics, keepdim=keepdim)
            stats = list(map(func, y, x))
            return stats if keepdim else reduce(add, stats)

    def weighted_statistics(self, y, x, weights, keepdim=False):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            y, x, weights = y[idx], x[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            yxT = np.einsum('nk,n,nh->nkh', y, weights, x)
            xxT = np.einsum('nk,n,nh->nkh', x, weights, x)
            yyT = np.einsum('nk,n,nh->nkh', y, weights, y)
            n = weights

            if not keepdim:
                yxT = np.sum(yxT, axis=0)
                xxT = np.sum(xxT, axis=0)
                yyT = np.sum(yyT, axis=0)
                n = np.sum(n, axis=0)

            return Stats([yxT, xxT, yyT, n])
        else:
            func = partial(self.weighted_statistics, keepdim=keepdim)
            stats = list(map(func, y, x, weights))
            return stats if keepdim else reduce(add, stats)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.drow / 2.)

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return - np.sum(np.log(np.diag(self.lmbda_chol)))

    def entropy(self):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, y, x, weights=None):
        stats = self.statistics(y, x) if weights is None\
            else self.weighted_statistics(y, x, weights)

        yxT, xxT, yyT, n = stats
        self.A = np.linalg.solve(xxT, yxT.T).T
        _sigma = (yyT - self.A.dot(yxT.T)) / n

        # numerical stabilization
        _sigma = symmetrize(_sigma) + 1e-16 * np.eye(self.drow)
        assert np.allclose(_sigma, _sigma.T)
        assert np.all(np.linalg.eigvalsh(_sigma) > 0.)

        self.lmbda = invpd(_sigma)

        return self
