import numpy as np
import numpy.random as npr
from numpy.core.umath_tests import inner1d
from scipy import linalg

from mimo.abstractions import Distribution
from mimo.util.general import inv_psd, blockarray


class LinearGaussian(Distribution):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """
    def __init__(self, A=None, sigma=None, affine=False):

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
    def din(self):
        # input dimension
        return self.A.shape[1]

    @property
    def dout(self):
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
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    def rvs(self, x=None, size=None):
        size = 1 if size is None else size
        A, sigma_chol = self.A, self.sigma_chol

        if self.affine:
            A, b = A[:, :-1], A[:, -1]

        x = npr.normal(size=(size, A.shape[1])) if x is None else x
        y = self.predict(x)
        y += npr.normal(size=(x.shape[0], self.dout)).dot(sigma_chol.T)

        return np.hstack((x, y))

    def predict(self, x):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:, :-1], A[:, -1]
            y = x.dot(A.T) + b.T
        else:
            y = x.dot(A.T)

        return y

    def mean(self):
        return self.A

    def mode(self):
        return self.A

    # distribution
    def log_likelihood(self, xy):
        A, sigma, dout = self.A, self.sigma, self.dout
        x, y = (xy[:, :-dout], xy[:, -dout:])

        if self.affine:
            A, b = A[:, :-1], A[:, -1]

        sigma_inv, L = inv_psd(sigma, return_chol=True)
        parammat = - 0.5 * blockarray([[A.T.dot(sigma_inv).dot(A),
                                        -A.T.dot(sigma_inv)],
                                       [-sigma_inv.dot(A), sigma_inv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum(contract, xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-dout, :-dout]), x)
            out += np.einsum(contract, y.dot(parammat[-dout:, -dout:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-dout, -dout:]), y)

        out -= 0.5 * dout * np.log(2. * np.pi) + np.log(np.diag(L)).sum()

        if self.affine:
            out += y.dot(sigma_inv).dot(b)
            out -= x.dot(A.T).dot(sigma_inv).dot(b)
            out -= 0.5 * b.dot(sigma_inv).dot(b)

        return out

    def log_partition(self):
        return 0.5 * self.dout * np.log(2. * np.pi)\
               + np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * (self.dout * np.log(2. * np.pi) + self.dout
                      + 2. * np.sum(np.log(np.diag(self.sigma_chol))))


class LinearGaussianWithNoisyInputs(Distribution):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """
    def __init__(self, mu=None, sigma_niw=None, A=None, sigma=None, affine=False):

        self.mu = mu
        self._sigma_niw = sigma_niw

        self.A = A
        self._sigma = sigma
        self.affine = affine

    @property
    def params(self):
        return self.mu, self.sigma_niw, self.A, self.sigma

    # @property
    # def params(self):
    #     return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma_niw, self.A, self.sigma = values

    # @params.setter
    # def params(self, values):
    #     self.mu, self.sigma = values

    @property
    def din(self):
        # input dimension
        return self.A.shape[1] if not self.affine else self.A.shape[1] + 1

    @property
    def dout(self):
        # output dimension
        return self.A.shape[0]

    @property
    def dim(self):
        return self.mu.shape[0]

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_niw(self):
        return self._sigma_niw

    # @property
    # def sigma(self):
    #     return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._sigma_chol = None

    @sigma_niw.setter
    def sigma_niw(self, value):
        self._sigma_niw = value
        self._sigma_niw_chol = None

    # @sigma.setter
    # def sigma(self, value):
    #     self._sigma = value
    #     self._sigma_chol = None

    @property
    def sigma_chol(self):
        if not hasattr(self, '_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    @property
    def sigma_niw_chol(self):
        if not hasattr(self, '_sigma_niw_chol') or self._sigma_niw_chol is None:
            self._sigma_niw_chol = np.linalg.cholesky(self.sigma_niw)
        return self._sigma_niw_chol

    # @property
    # def sigma_chol(self):
    #     if not hasattr(self, '_sigma_chol') or self._sigma_chol is None:
    #         self._sigma_chol = np.linalg.cholesky(self.sigma)
    #     return self._sigma_chol

    def rvs(self, x=None, size=None):
        size = 1 if size is None else size
        A, sigma_chol = self.A, self.sigma_chol

        if self.affine:
            A, b = A[:, :-1], A[:, -1]

        # x = npr.normal(size=(size, A.shape[1])) if x is None else x       #Fixme add the part 'if x is None else x to' code below
        if size is None:
            x = self.mu + npr.normal(size=self.dim).dot(self.sigma_niw_chol.T)
        else:
            size = tuple([size, self.dim])
            x = self.mu + npr.normal(size=size).dot(self.sigma_niw_chol.T)

        y = self.predict(x)
        y += npr.normal(size=(x.shape[0], self.dout)).dot(sigma_chol.T)

        return np.hstack((x, y))

    # def rvs(self, size=None):
    #     if size is None:
    #         return self.mu + npr.normal(size=self.dim).dot(self.sigma_chol.T)
    #     else:
    #         size = tuple([size, self.dim])
    #         return self.mu + npr.normal(size=size).dot(self.sigma_chol.T)

    def predict(self, x):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:, :-1], A[:, -1]
            y = x.dot(A.T) + b.T
        else:
            y = x.dot(A.T)

        return y

    def mean(self):
        return self.mu, self.A

    # def mean(self):
    #     return self.mu

    def mode(self):
        return self.mu, self.A

    # def mode(self):
    #     return self.mu

    # distribution
    def log_likelihood(self, xy):
        # log-likelihood of linear gaussian
        A, sigma, dout = self.A, self.sigma, self.dout
        x, y = (xy[:, :-dout], xy[:, -dout:])

        if self.affine:
            A, b = A[:, :-1], A[:, -1]

        sigma_inv, L = inv_psd(sigma, return_chol=True)
        parammat = - 0.5 * blockarray([[A.T.dot(sigma_inv).dot(A),
                                        -A.T.dot(sigma_inv)],
                                       [-sigma_inv.dot(A), sigma_inv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum(contract, xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-dout, :-dout]), x)
            out += np.einsum(contract, y.dot(parammat[-dout:, -dout:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-dout, -dout:]), y)

        out -= 0.5 * dout * np.log(2. * np.pi) + np.log(np.diag(L)).sum()

        if self.affine:
            out += y.dot(sigma_inv).dot(b)
            out -= x.dot(A.T).dot(sigma_inv).dot(b)
            out -= 0.5 * b.dot(sigma_inv).dot(b)

        #log-likelihood of gaussian
        try:
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            xc = np.nan_to_num(x).reshape((-1, self.dim)) - self.mu
            xs = linalg.solve_triangular(self.sigma_niw_chol, xc.T, lower=True)
            out1 = - 0.5 * self.dim * np.log(2. * np.pi) -\
                  np.sum(np.log(np.diag(self.sigma_niw_chol))) - 0.5 * inner1d(xs.T, xs.T)
            out1[bads] = 0
            # return out
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density
            out1 = np.repeat(-np.inf, x.shape[0])

        return out + out1

    # def log_likelihood(self, x):
    #     try:
    #         bads = np.isnan(np.atleast_2d(x)).any(axis=1)
    #         xc = np.nan_to_num(x).reshape((-1, self.dim)) - self.mu
    #         xs = linalg.solve_triangular(self.sigma_chol, xc.T, lower=True)
    #         out = - 0.5 * self.dim * np.log(2. * np.pi) -\
    #               np.sum(np.log(np.diag(self.sigma_chol))) - 0.5 * inner1d(xs.T, xs.T)
    #         out[bads] = 0
    #         return out
    #     except np.linalg.LinAlgError:
    #         # NOTE: degenerate distribution doesn't have a density
    #         return np.repeat(-np.inf, x.shape[0])

    def log_partition(self):
        return 0.5 * self.dout * np.log(2. * np.pi) +\
               np.sum(np.log(np.diag(self.sigma_chol))) +\
               0.5 * self.dim * np.log(2. * np.pi) +\
               np.sum(np.log(np.diag(self.sigma_niw_chol)))

    # def log_partition(self):
    #     return 0.5 * self.dim * np.log(2. * np.pi) +\
    #            np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * (self.dout * np.log(2. * np.pi) + self.dout +
                      2. * np.sum(np.log(np.diag(self.sigma_chol)))) +\
               0.5 * (self.dim * np.log(2. * np.pi) + self.dim +\
                      2. * np.sum(np.log(np.diag(self.sigma_niw_chol))))

    # def entropy(self):
    #     return 0.5 * (self.dim * np.log(2. * np.pi) + self.dim +
    #                   2. * np.sum(np.log(np.diag(self.sigma_chol))))
