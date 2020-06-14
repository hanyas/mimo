import numpy as np
import numpy.random as npr

from scipy.special import gammaln, digamma

from mimo.abstraction import Distribution


class Gamma(Distribution):

    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def dim(self):
        return len(self.alphas)

    def rvs(self, size=1):
        # numpy uses a different parameterization
        return npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        return self.alphas / self.betas

    def mode(self):
        assert np.all(self.alphas >= 1.)
        return (self.alphas - 1.) / self.betas

    def log_likelihood(self, x):
        loglik = np.sum(- gammaln(self.alphas) + self.alphas * np.log(self.betas)
                        + (self.alphas - 1.) * np.log(x) - x * self.betas)
        return loglik

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def entropy(self):
        return np.sum(self.alphas - np.log(self.betas) +
                      gammaln(self.alphas) + (1. - self.alphas) * digamma(self.alphas))


class InverseGamma(Distribution):

    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def dim(self):
        return len(self.alphas)

    def rvs(self, size=1):
        # numpy uses a different parameterization
        return 1. / npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        assert np.all(self.alphas >= 1.)
        return self.betas / (self.alphas - 1)

    def mode(self):
        return self.betas / (self.alphas + 1.)

    def log_likelihood(self, x):
        loglik = np.sum(- gammaln(self.alphas)
                        + self.alphas * np.log(self.betas)
                        - (self.alphas + 1.) * np.log(x) - x / self.betas)
        return loglik

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def entropy(self):
        return np.sum(self.alphas + np.log(self.betas) +
                      gammaln(self.alphas) - (1. + self.alphas) * digamma(self.alphas))
