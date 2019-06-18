import numpy as np
import numpy.random as npr

from numpy.core.umath_tests import inner1d
from scipy import linalg

from mimo.abstractions import Distribution
from mimo.util.general import flattendata


class Gaussian(Distribution):

    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self._sigma = sigma

    @property
    def params(self):
        return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma = values

    @property
    def dim(self):
        return self.mu.shape[0]

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

    def rvs(self, size=None):
        if size is None:
            return self.mu + npr.normal(size=self.dim).dot(self.sigma_chol.T)
        else:
            size = tuple([size, self.dim])
            return self.mu + npr.normal(size=size).dot(self.sigma_chol.T)

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def log_likelihood(self, x):
        try:
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            xc = np.nan_to_num(x).reshape((-1, self.dim)) - self.mu
            xs = linalg.solve_triangular(self.sigma_chol, xc.T, lower=True)
            out = - 0.5 * self.dim * np.log(2. * np.pi) -\
                  np.sum(np.log(np.diag(self.sigma_chol))) - 0.5 * inner1d(xs.T, xs.T)
            out[bads] = 0
            return out
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density
            return np.repeat(-np.inf, x.shape[0])

    def log_partition(self):
        return 0.5 * self.dim * np.log(2. * np.pi) +\
               np.sum(np.log(np.diag(self.sigma_chol)))

    def entropy(self):
        return 0.5 * (self.dim * np.log(2. * np.pi) + self.dim +
                      2. * np.sum(np.log(np.diag(self.sigma_chol))))

    _parameterplot = None
    _scatterplot = None

    def plot(self, ax=None, data=None, color='b', label='', alpha=1., update=False, draw=True):

        import matplotlib.pyplot as plt
        from mimo.util.plot import plot_gaussian

        ax = ax if ax else plt.gca()

        if data is not None:
            data = flattendata(data)
            self._scatterplot = ax.scatter(data[:, 0], data[:, 1], marker='.', color=color)

        self._parameterplot = plot_gaussian(self.mu, self.sigma, color=color, label=label,
                                            alpha=min(1 - 1e-3, alpha), ax=ax,
                                            artists=self._parameterplot if update else None)
        if draw:
            plt.draw()

        return [self._scatterplot] + list(self._parameterplot)


class DiagonalGaussian(Gaussian):

    def __init__(self, mu=None, sigmas=None):
        self._sigmas = sigmas
        super(DiagonalGaussian, self).__init__(mu=mu, sigma=self.sigma)

    @property
    def sigma(self):
        if self._sigmas is not None:
            return np.diag(self._sigmas)
        else:
            return None

    @sigma.setter
    def sigma(self, value):
        value = np.array(value)
        if value.ndim == 1:
            self._sigmas = value
        else:
            self._sigmas = np.diag(value)
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if not hasattr(self, '_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.diag(np.sqrt(self._sigmas))
        return self._sigma_chol


if __name__ == "__main__":
    pass
