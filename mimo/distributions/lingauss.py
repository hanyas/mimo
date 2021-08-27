import numpy as np
import numpy.random as npr

import scipy as sc

from operator import add
from functools import reduce, partial

from mimo.utils.abstraction import Statistics as Stats
from mimo.utils.matrix import symmetrize


class LinearGaussianWithPrecision:

    def __init__(self, column_dim, row_dim,
                 A=None, lmbda=None, affine=True):

        self.column_dim = column_dim
        self.row_dim = row_dim

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
    def nb_params(self):
        return self.column_dim * self.row_dim \
               + self.row_dim * (self.row_dim + 1) / 2

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

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

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('dl,...l->...d', A, x, optimize=True) + b
        else:
            y = np.einsum('dl,...l->...d', self.A, x, optimize=True)
        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        size = self.output_dim if x.ndim == 1 else tuple([x.shape[0], self.output_dim])
        return self.mean(x) + npr.normal(size=size).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.output_dim / 2.)

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y, fold=True):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if fold:
                contract = 'nd,nl->dl'
                n = y.shape[0]
            else:
                contract = 'nd,nl->ndl'
                n = np.ones((y.shape[0], ))

            yxT = np.einsum(contract, y, x, optimize=True)
            xxT = np.einsum(contract, x, x, optimize=True)
            yyT = np.einsum(contract, y, y, optimize=True)

            return Stats([yxT, xxT, yyT, n])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, x, y))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,n,nl->dl'
            n = np.sum(weights)

            yxT = np.einsum(contract, y, weights, x, optimize=True)
            xxT = np.einsum(contract, x, weights, x, optimize=True)
            yyT = np.einsum(contract, y, weights, y, optimize=True)

            return Stats([yxT, xxT, yyT, n])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True)\
                      - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nd,dl,nl->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxT, xxT, yyT, n = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        self.A = np.linalg.solve(xxT, yxT.T).T
        sigma = (yyT - self.A.dot(yxT.T)) / n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.lmbda = np.linalg.inv(sigma)


class StackedLinearGaussiansWithPrecision:

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas=None, affine=True):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas = [None] * self.size if lmbdas is None else lmbdas
        self.dists = [LinearGaussianWithPrecision(column_dim, row_dim,
                                                  As[k], lmbdas[k], affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas

    @params.setter
    def params(self, values):
        self.As, self.lmbdas = values

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @lmbdas.setter
    def lmbdas(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda = value[k]

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->k...d', As, x, optimize=True) + bs[:, None, :]
        else:
            y = np.einsum('kdl,...l->k...d', self.As, x, optimize=True)
        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y, fold=True):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if fold:
                contract = 'nd,nl->dl'
                n = y.shape[0]
            else:
                contract = 'nd,nl->ndl'
                n = np.ones((y.shape[0], ))

            yxT = np.einsum(contract, y, x, optimize=True)
            xxT = np.einsum(contract, x, x, optimize=True)
            yyT = np.einsum(contract, y, y, optimize=True)

            yxTk = np.array([yxT for _ in range(self.size)])
            xxTk = np.array([xxT for _ in range(self.size)])
            yyTk = np.array([yyT for _ in range(self.size)])
            nk = np.array([n for _ in range(self.size)])

            return Stats([yxTk, xxTk, yyTk, nk])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, x, y))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[:, idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            contract = 'nd,kn,nl->kdl'

            yxTk = np.einsum(contract, y, weights, x, optimize=True)
            xxTk = np.einsum(contract, x, weights, x, optimize=True)
            yyTk = np.einsum(contract, y, weights, y, optimize=True)
            nk = np.sum(weights, axis=1)

            return Stats([yxTk, xxTk, yyTk, nk])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists])

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('knd,kdl,nl->kn', mu, self.lmbdas, y, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->kn', y, self.lmbdas, y, optimize=True)

            log_lik[:, bads] = 0.
            log_lik += - self.log_partition(x)\
                       + np.expand_dims(self.log_base(), axis=1)
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, yyTk, nk = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        lmbdas = np.zeros((self.size, self.output_dim, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma = (yyTk[k] - As[k].dot(yxTk[k].T)) / nk[k]

            # numerical stabilization
            sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
            assert np.allclose(sigma, sigma.T)
            assert np.all(np.linalg.eigvalsh(sigma) > 0.)

            lmbdas[k] = np.linalg.inv(sigma)

        self.As = As
        self.lmbdas = lmbdas


class TiedLinearGaussiansWithPrecision(StackedLinearGaussiansWithPrecision):

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas=None, affine=True):

        super(TiedLinearGaussiansWithPrecision, self).__init__(size, column_dim, row_dim,
                                                               As, lmbdas, affine)

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, yyT, n = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        sigma = np.zeros((self.output_dim, self.output_dim))

        sigma = yyT
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma -= As[k].dot(yxTk[k].T)
        sigma /= n

        # numerical stabilization
        sigma = symmetrize(sigma) + 1e-16 * np.eye(self.output_dim)
        assert np.allclose(sigma, sigma.T)
        assert np.all(np.linalg.eigvalsh(sigma) > 0.)

        self.As = As
        lmbda = np.linalg.inv(sigma)
        self.lmbdas = np.array(self.size * [lmbda])


class LinearGaussianWithDiagonalPrecision:

    def __init__(self, column_dim, row_dim,
                 A=None, lmbda_diag=None, affine=True):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.A = A
        self.affine = affine

        self._lmbda_diag = lmbda_diag
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def params(self):
        return self.A, self.lmbda_diag

    @params.setter
    def params(self, values):
        self.A, self.lmbda_diag = values

    @property
    def nb_params(self):
        return self.column_dim * self.row_dim + self.row_dim

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

    @property
    def lmbda_diag(self):
        return self._lmbda_diag

    @lmbda_diag.setter
    def lmbda_diag(self, value):
        self._lmbda_diag = value
        self._lmbda_chol = None
        self._lmbda_chol_inv = None

    @property
    def lmbda(self):
        assert self._lmbda_diag is not None
        return np.diag(self._lmbda_diag)

    @property
    def lmbda_chol(self):
        if self._lmbda_chol is None:
            self._lmbda_chol = np.diag(np.sqrt(self.lmbda_diag))
        return self._lmbda_chol

    @property
    def lmbda_chol_inv(self):
        if self._lmbda_chol_inv is None:
            self._lmbda_chol_inv = np.diag(1. / np.sqrt(self.lmbda_diag))
        return self._lmbda_chol_inv

    @property
    def sigma_diag(self):
        return 1. / self.lmbda_diag

    @property
    def sigma(self):
        return np.diag(self.sigma_diag)

    def predict(self, x):
        if self.affine:
            A, b = self.A[:, :-1], self.A[:, -1]
            y = np.einsum('dl,...l->...d', A, x, optimize=True) + b.T
        else:
            y = np.einsum('dl,...l->...d', self.A, x, optimize=True)

        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        size = self.output_dim if x.ndim == 1 else tuple([x.shape[0], self.output_dim])
        return self.mean(x) + npr.normal(size=size).dot(self.lmbda_chol_inv.T)

    @property
    def base(self):
        return np.power(2. * np.pi, - self.output_dim / 2.)

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y, fold=True):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if fold:
                c0, c1 = 'nd,nl->dl', 'nd,nd->d'
                nd = np.broadcast_to(y.shape[0], (self.output_dim,))
            else:
                c0, c1 = 'nd,nl->ndl', 'nd,nd->nd'
                nd = np.ones((y.shape[0], self.output_dim))

            xxT = np.einsum(c0, x, x, optimize=True)
            yxT = np.einsum(c0, y, x, optimize=True)
            yy = np.einsum(c1, y, y, optimize=True)

            return Stats([yxT, xxT, nd, yy])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, x, y))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxT = np.einsum('nd,n,nl->dl', x, weights, x, optimize=True)
            yxT = np.einsum('nd,n,nl->dl', y, weights, x, optimize=True)
            yy = np.einsum('nd,n,nd->d', y, weights, y, optimize=True)
            nd = np.broadcast_to(np.sum(weights), (self.output_dim, ))

            return Stats([yxT, xxT, nd, yy])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        mu = self.predict(x)
        return 0.5 * np.einsum('nd,dl,nl->n', mu, self.lmbda, mu)\
               - np.sum(np.log(np.diag(self.lmbda_chol)))

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('nd,dl,nl->n', mu, self.lmbda, y, optimize=True) \
                      - 0.5 * np.einsum('nd,dl,nl->n', y, self.lmbda, y, optimize=True)

            log_lik[bads] = 0.
            log_lik += - self.log_partition(x) + self.log_base()
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights=None):
        yxT, xxT, nd, yy = self.statistics(x, y) if weights is None\
            else self.weighted_statistics(x, y, weights)

        self.A = np.linalg.solve(xxT, yxT.T).T
        sigmas = (yy - np.einsum('dl,dl->d', self.A, yxT)) / nd
        self.lmbda_diag = 1. / sigmas


class StackedLinearGaussiansWithDiagonalPrecision:

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas_diags=None, affine=True):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        self.affine = affine

        As = [None] * self.size if As is None else As
        lmbdas_diags = [None] * self.size if lmbdas_diags is None else lmbdas_diags
        self.dists = [LinearGaussianWithDiagonalPrecision(column_dim, row_dim,
                                                          As[k], lmbdas_diags[k],
                                                          affine=affine)
                      for k in range(self.size)]

    @property
    def params(self):
        return self.As, self.lmbdas_diags

    @params.setter
    def params(self, values):
        self.As, self.lmbdas_diags = values

    @property
    def input_dim(self):
        return self.column_dim - 1 if self.affine\
               else self.column_dim

    @property
    def output_dim(self):
        return self.row_dim

    @property
    def As(self):
        return np.array([dist.A for dist in self.dists])

    @As.setter
    def As(self, value):
        for k, dist in enumerate(self.dists):
            dist.A = value[k]

    @property
    def lmbdas_diags(self):
        return np.array([dist.lmbda_diag for dist in self.dists])

    @lmbdas_diags.setter
    def lmbdas_diags(self, value):
        for k, dist in enumerate(self.dists):
            dist.lmbda_diag = value[k]

    @property
    def lmbdas(self):
        return np.array([dist.lmbda for dist in self.dists])

    @property
    def lmbdas_chol(self):
        return np.array([dist.lmbda_chol for dist in self.dists])

    @property
    def lmbdas_chol_inv(self):
        return np.array([dist.lmbda_chol_inv for dist in self.dists])

    @property
    def sigmas_diag(self):
        return np.array([dist.sigma_diag for dist in self.dists])

    @property
    def sigmas(self):
        return np.array([dist.sigma for dist in self.dists])

    def predict(self, x):
        if self.affine:
            As, bs = self.As[:, :, :-1], self.As[:, :, -1]
            y = np.einsum('kdl,...l->k...d', As, x, optimize=True) + bs[:, None, :]
        else:
            y = np.einsum('kdl,...l->k...d', self.As, x, optimize=True)
        return y

    def mean(self, x):
        return self.predict(x)

    def mode(self, x):
        return self.predict(x)

    def rvs(self, x):
        return np.array([dist.rvs(x) for dist in self.dists])

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def statistics(self, x, y, fold=True):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y = x[idx], y[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            if fold:
                c0, c1 = 'nd,nl->dl', 'nd,nd->d'
                nd = np.broadcast_to(y.shape[0], (self.output_dim,))
            else:
                c0, c1 = 'nd,nl->ndl', 'nd,nd->nd'
                nd = np.ones((y.shape[0], self.output_dim))

            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)
            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)
            yy = np.einsum('nd,nd->d', y, y, optimize=True)

            xxTk = np.array([xxT for _ in range(self.size)])
            yxTk = np.array([yxT for _ in range(self.size)])
            yyk = np.array([yy for _ in range(self.size)])
            ndk = np.array([nd for _ in range(self.size)])

            return Stats([yxTk, xxTk, ndk, yyk])
        else:
            func = partial(self.statistics, fold=fold)
            stats = list(map(func, x, y))
            return reduce(add, stats) if fold else stats

    def weighted_statistics(self, x, y, weights):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            idx = np.logical_and(~np.isnan(x).any(axis=1),
                                 ~np.isnan(y).any(axis=1))
            x, y, weights = x[idx], y[idx], weights[idx]

            if self.affine:
                x = np.hstack((x, np.ones((x.shape[0], 1))))

            xxTk = np.einsum('nd,nk,nl->kdl', x, weights, x, optimize=True)
            yxTk = np.einsum('nd,nk,nl->kdl', y, weights, x, optimize=True)
            yyk = np.einsum('nd,nk,nd->kd', y, weights, y, optimize=True)
            ndk = np.broadcast_to(np.sum(weights, axis=0, keepdims=True),
                                  (self.size, self.output_dim))

            return Stats([yxTk, xxTk, ndk, yyk])
        else:
            stats = list(map(self.weighted_statistics, x, y, weights))
            return reduce(add, stats)

    def log_partition(self, x):
        return np.array([dist.log_partition(x) for dist in self.dists]).T

    def log_likelihood(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            bads = np.logical_and(np.isnan(np.atleast_2d(x)).any(axis=1),
                                  np.isnan(np.atleast_2d(y)).any(axis=1))

            x = np.nan_to_num(x, copy=False).reshape((-1, self.input_dim))
            y = np.nan_to_num(y, copy=False).reshape((-1, self.output_dim))

            mu = self.mean(x)
            log_lik = np.einsum('knd,kdl,nl->kn', mu, self.lmbdas, y, optimize=True)\
                      - 0.5 * np.einsum('nd,kdl,nl->kn', y, self.lmbdas, y, optimize=True)

            log_lik[:, bads] = 0.
            log_lik += - self.log_partition(x)\
                       + np.expand_dims(self.log_base(), axis=1)
            return log_lik
        else:
            return list(map(self.log_likelihood, x, y))

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, ndk, yyk = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        lmbdas = np.zeros((self.size, self.output_dim))
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigmas = (yyk[k] - np.einsum('dl,dl->d', As[k], yxTk[k])) / ndk[k]
            lmbdas[k] = 1. / sigmas

        self.As = As
        self.lmbdas_diags = lmbdas


class TiedLinearGaussiansWithDiagonalPrecision(StackedLinearGaussiansWithDiagonalPrecision):

    def __init__(self, size, column_dim, row_dim,
                 As=None, lmbdas_diags=None, affine=True):

        super(TiedLinearGaussiansWithDiagonalPrecision, self).__init__(size, column_dim, row_dim,
                                                                       As, lmbdas_diags, affine)

    # Max likelihood
    def max_likelihood(self, x, y, weights):
        yxTk, xxTk, nd, yy = self.weighted_statistics(x, y, weights)

        As = np.zeros((self.size, self.column_dim, self.row_dim))
        sigma_diag = np.zeros((self.output_dim, ))

        sigma_diag = yy
        for k in range(self.size):
            As[k] = np.linalg.solve(xxTk[k], yxTk[k].T).T
            sigma_diag -= np.einsum('dl,dl->d', As[k], yxTk[k])
        sigma_diag /= nd

        self.As = As
        lmbda_diag = 1. / sigma_diag
        self.lmbdas_diags = np.array(self.size * [lmbda_diag])
