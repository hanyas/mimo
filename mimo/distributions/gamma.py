import numpy as np
import numpy.random as npr

from scipy.special import gammaln, digamma

from mimo.abstractions import Distribution

import warnings


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

    def rvs(self, size=None):
        # numpy uses a different parameterization
        return npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        return self.alphas / self.betas

    def mode(self):
        if np.all(self.alphas >= 1.):
            return (self.alphas - 1.) / self.betas
        else:
            warnings.warn("Mode of Gamma distribution not defined")
            return None

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

    def rvs(self, size=None):
        # numpy uses a different parameterization
        return 1. / npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        if np.all(self.alphas >= 1.):
            return self.betas / (self.alphas - 1)
        else:
            warnings.warn("Mean of Inverse Gamma distribution not defined")
            return None

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
