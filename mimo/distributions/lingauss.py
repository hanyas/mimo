import numpy as np
import numpy.random as npr

from mimo.abstractions import Distribution
from mimo.util.general import inv_psd, blockarray


class LinearGaussian(Distribution):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """
    def __init__(self,  A=None, sigma=None, affine=False,):

        self.A = A
        self._sigma = sigma
        self.affine = affine

    @property
    def params(self):
        return self.A, self.sigma

    @params.setter
    def params(self, values):
        self.A, self.sigma = values

    @property
    def din(self):
        # input dimension
        return self.A.shape[1] if not self.affine else self.A.shape[1] + 1

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
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if not hasattr(self, '_sigma_chol') or self._sigma_chol is None:
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
        return 0.5 * self.dout * np.log(2. * np.pi) +\
               np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * (self.dout * np.log(2. * np.pi) + self.dout +
                      2. * np.sum(np.log(np.diag(self.sigma_chol))))
