import numpy as np
import numpy.random as npr
from numpy.core.umath_tests import inner1d
from scipy import linalg

from mimo.abstractions import Distribution
from mimo.util.general import inv_psd, blockarray
from mimo.util.general import near_pd


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
        # input dimension, intercept excluded
        if self.affine:
            return self.A.shape[1] - 1
        else:
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
            self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
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
        return 0.5 * self.dout * np.log(2. * np.pi) + self.dout\
               + np.sum(np.log(np.diag(self.sigma_chol)))


class LinearGaussianWithNoisyInputs(Distribution):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    The input is modelled as a Multivariate Gaussian distribution
    Parameters are a constant mean and covariance matrix:
        mu, sigma
    """
    def __init__(self, mu=None, sigma_niw=None, A=None, sigma=None, affine=False):

        self.mu = mu
        self._sigma_niw = sigma_niw
        self._sigma_niw_chol = None

        self.A = A
        self._sigma = sigma
        self._sigma_chol = None

        self.affine = affine

    @property
    def params(self):
        return self.mu, self.sigma_niw, self.A, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma_niw, self.A, self.sigma = values

    @property
    def num_parameters(self):
        _num_out = self.dout + self.dout * (self.dout + 1) / 2
        _num_in = self.din + self.din * (self.din + 1) / 2
        _num = _num_out + _num_in
        if self.affine:
            _num += self.dout
        return _num

    @property
    def din(self):
        # input dimension, intercept excluded
        if self.affine:
            return self.A.shape[1] - 1
        else:
            return self.A.shape[1]

    @property
    def dout(self):
        # output dimension
        return self.A.shape[0]

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_niw(self):
        return self._sigma_niw

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._sigma_chol = None

    @sigma_niw.setter
    def sigma_niw(self, value):
        self._sigma_niw = value
        self._sigma_niw_chol = None

    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
        return self._sigma_chol

    @property
    def sigma_niw_chol(self):
        if self._sigma_niw_chol is None:
            self._sigma_niw_chol = np.linalg.cholesky(near_pd(self.sigma_niw))
        return self._sigma_niw_chol

    def rvs(self, x=None, size=None):
        size = 1 if size is None else size
        _, sigma_chol = self.A, self.sigma_chol

        if size is None:
            x = self.mu + npr.normal(size=self.din).dot(self.sigma_niw_chol.T) if x is None else x
        else:
            size = tuple([size, self.din])
            x = self.mu + npr.normal(size=size).dot(self.sigma_niw_chol.T) if x is None else x

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
        return self.mu, self.A

    def mode(self):
        return self.mu, self.A

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
            tmp = np.einsum(contract, xy.dot(parammat), xy)
        else:
            tmp = np.einsum(contract, x.dot(parammat[:-dout, :-dout]), x)
            tmp += np.einsum(contract, y.dot(parammat[-dout:, -dout:]), y)
            tmp += 2. * np.einsum(contract, x.dot(parammat[:-dout, -dout:]), y)

        tmp -= 0.5 * dout * np.log(2. * np.pi) + np.sum(np.log(np.diag(L)))

        if self.affine:
            tmp += y.dot(sigma_inv).dot(b)
            tmp -= x.dot(A.T).dot(sigma_inv).dot(b)
            tmp -= 0.5 * b.dot(sigma_inv).dot(b)

        # log-likelihood of gaussian
        try:
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            xc = np.nan_to_num(x).reshape((-1, self.din)) - self.mu
            xs = linalg.solve_triangular(self.sigma_niw_chol, xc.T, lower=True)
            aux = - 0.5 * self.din * np.log(2. * np.pi)\
                  - np.sum(np.log(np.diag(self.sigma_niw_chol)))\
                  - 0.5 * inner1d(xs.T, xs.T)
            aux[bads] = 0
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density
            aux = np.repeat(-np.inf, x.shape[0])

        return tmp + aux

    def log_partition(self):
        return 0.5 * self.dout * np.log(2. * np.pi)\
               + np.sum(np.log(np.diag(self.sigma_chol)))\
               + 0.5 * self.din * np.log(2. * np.pi)\
               + np.sum(np.log(np.diag(self.sigma_niw_chol)))

    def entropy(self):
        return 0.5 * (self.dout * np.log(2. * np.pi) + self.dout
                      + 2. * np.sum(np.log(np.diag(self.sigma_chol))))\
               + 0.5 * (self.din * np.log(2. * np.pi) + self.din
                        + 2. * np.sum(np.log(np.diag(self.sigma_niw_chol))))
