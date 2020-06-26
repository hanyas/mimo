import numpy as np
import numpy.random as npr

from functools import reduce

from mimo.abstraction import Conditional
from mimo.abstraction import Statistics as Stats

from mimo.util.matrix import nearpd, blockarray, invpd


class LinearGaussian(Conditional):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """

    def __init__(self, A=None, sigma=None, affine=True):

        self.A = A
        self.affine = affine

        self._sigma = sigma
        self._sigma_chol = None

    @property
    def params(self):
        return self.A, self.sigma

    @params.setter
    def params(self, values):
        self.A, self.sigma = values

    @property
    def nb_params(self):
        return self.dcol * self.drow + self.drow * (self.drow + 1) / 2

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
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        # reset Cholesky for new values of sigma
        # A new Cholesky will be computed when needed
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(nearpd(self.sigma))
        return self._sigma_chol

    def rvs(self, x=None):
        assert x is not None
        size = 1 if x.ndim == 1 else x.shape[0]

        y = self.predict(x)
        y += npr.normal(size=(size, self.drow)).dot(self.sigma_chol.T)

        return y

    def predict(self, x):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:, :-1], A[:, -1]
            y = x.dot(A.T) + b.T
        else:
            y = x.dot(A.T)

        return y

    def mean(self, x):
        return self.A @ x

    def mode(self, x):
        return self.A @ x

    # distribution
    def log_likelihood(self, y, x=None):
        A, sigma, drow = self.A, self.sigma, self.drow
        xy = np.hstack((x, y))

        if self.affine:
            A, b = A[:, :-1], A[:, -1]

        sigma_inv, L = invpd(sigma, return_chol=True)
        parammat = - 0.5 * blockarray([[A.T.dot(sigma_inv).dot(A),
                                        -A.T.dot(sigma_inv)],
                                       [-sigma_inv.dot(A), sigma_inv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum(contract, xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-drow, :-drow]), x)
            out += np.einsum(contract, y.dot(parammat[-drow:, -drow:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-drow, -drow:]), y)

        out -= 0.5 * drow * np.log(2. * np.pi) + np.log(np.diag(L)).sum()

        if self.affine:
            out += y.dot(sigma_inv).dot(b)
            out -= x.dot(A.T).dot(sigma_inv).dot(b)
            out -= 0.5 * b.dot(sigma_inv).dot(b)

        return out

    def log_partition(self):
        return 0.5 * self.drow * np.log(2. * np.pi)\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * self.drow * np.log(2. * np.pi) + self.drow\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def get_statistics(self, y, x):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(1),
                                 ~np.isnan(x).any(1))
            y, x = y[idx], x[idx]
            n, drow, dcol = y.shape[0], self.drow, self.dcol

            data = np.hstack((x, y))
            stats = data.T.dot(data)
            xxT, yxT, yyT = stats[:-drow, :-drow], stats[-drow:, :-drow], stats[-drow:, -drow:]

            if self.affine:
                xy = np.sum(data, axis=0)
                x, y = xy[:-drow], xy[-drow:]
                xxT = blockarray([[xxT, x[:, np.newaxis]],
                                  [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return Stats([yxT, xxT, yyT, n])
        else:
            return reduce(lambda a, b: a + b, list(map(self.get_statistics, y, x)))

    def get_weighted_statistics(self, y, x, weights):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            idx = np.logical_and(~np.isnan(y).any(1),
                                 ~np.isnan(x).any(1))
            y, x, weights = y[idx], x[idx], weights[idx]
            n, drow, dcol = weights.sum(), self.drow, self.dcol

            data = np.hstack((x, y))
            stats = data.T.dot(weights[:, np.newaxis] * data)
            xxT, yxT, yyT = stats[:-drow, :-drow], stats[-drow:, :-drow], stats[-drow:, -drow:]

            if self.affine:
                xy = weights.dot(data)
                x, y = xy[:-drow], xy[-drow:]
                xxT = blockarray([[xxT, x[:, np.newaxis]], [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return Stats([yxT, xxT, yyT, n])
        else:
            return reduce(lambda a, b: a + b, list(map(self.get_weighted_statistics, y, x, weights)))

    def _empty_statistics(self):
        return Stats([np.zeros((self.drow, self.dcol)),
                      np.zeros((self.dcol, self.dcol)),
                      np.zeros((self.drow, self.drow)), 0])

    # Max likelihood
    def max_likelihood(self, y, x, weights=None):
        stats = self.posterior.statistics(y, x) if weights is None\
            else self.posterior.weighted_statistics(y, x, weights)

        # (yxT, xxT, yyT, n)
        yxT, xxT, yyT, n = stats

        self.A = np.linalg.solve(xxT, yxT.T).T
        self.sigma = (yyT - self.A.dot(yxT.T)) / n

        def symmetrize(A):
            return (A + A.T) / 2.

        # numerical stabilization
        # self.sigma = near_pd(symmetrize(self.sigma) + 1e-16 * np.eye(self.drow))
        self.sigma = symmetrize(self.sigma) + 1e-16 * np.eye(self.drow)

        assert np.allclose(self.sigma, self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        return self
