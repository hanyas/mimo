import numpy as np
import numpy.random as npr

from mimo.distribution import Distribution, Conditional
from mimo.distributions.gaussian import Gaussian

from mimo.util.general import inv_psd, blockarray
from mimo.util.general import near_pd


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
            self._sigma_chol = np.linalg.cholesky(near_pd(self.sigma))
        return self._sigma_chol

    def rvs(self, x=None):
        assert x is not None
        size = 1 if x.ndim == 1 else x.shape[0]

        y = self.predict(x)
        y += npr.normal(size=(size, self.drow)).dot(self.sigma_chol.T)

        return np.hstack((x, y))

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

        sigma_inv, L = inv_psd(sigma, return_chol=True)
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


class JointLinearGaussian(Distribution):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    The input is modelled as a Multivariate Gaussian distribution
    Parameters are a constant mean and covariance matrix:
        mu, sigma
    """

    def __init__(self, mu=None, sigma_in=None,
                 A=None, sigma_out=None, affine=True):

        self.gaussian = Gaussian(mu=mu, sigma=sigma_in)
        self.linear_gaussian = LinearGaussian(A=A, sigma=sigma_out, affine=affine)

    @property
    def params(self):
        return self.gaussian.params, self.linear_gaussian.params

    @params.setter
    def params(self, values):
        self.gaussian.params = values[:2]
        self.linear_gaussian.params = values[2:]

    @property
    def nb_params(self):
        return self.gaussian.nb_params\
               + self.linear_gaussian.nb_params

    @property
    def dcol(self):
        return self.linear_gaussian.dcol

    @property
    def drow(self):
        return self.linear_gaussian.drow

    def rvs(self, size=1):
        size = 1 if size is None else size
        x = self.gaussian.rvs(size=size)
        xy = self.linear_gaussian.rvs(x=x)
        return xy

    def predict(self, x):
        return self.linear_gaussian.predict(x)

    def mean(self):
        x = self.gaussian.mean()
        y = self.linear_gaussian.mean(x)
        return x, y

    def mode(self):
        x = self.gaussian.mode()
        y = self.linear_gaussian.mode(x)
        return x, y

    # distribution
    def log_likelihood(self, xy):
        x = xy[:, :-self.drow]
        y = xy[:, self.dcol:]

        # log-likelihood of linear gaussian
        tmp = self.linear_gaussian.log_likelihood(y, x)

        # log-likelihood of gaussian
        aux = self.gaussian.log_likelihood(x)

        return tmp + aux

    def log_partition(self):
        return self.linear_gaussian.log_partition()\
               + self.gaussian.log_partition()

    def entropy(self):
        return self.linear_gaussian.entropy() + self.gaussian.entropy()
