import numpy as np
import numpy.random as npr
from scipy.special import digamma, gammaln

from mimo.abstraction import Distribution

import warnings


class Dirichlet(Distribution):

    def __init__(self, K=None, alphas=None):
        self.K = K
        self.alphas = alphas

    @property
    def params(self):
        return self.alphas

    @params.setter
    def params(self, values):
        self.alphas = values

    @property
    def dim(self):
        return self.K

    def rvs(self, size=1):
        size = None if size == 1 else size
        return npr.dirichlet(self.alphas, size)

    def mean(self):
        return self.alphas / np.sum(self.alphas)

    def mode(self):
        assert np.all(self.alphas > 1.), "Make sure alphas > 1."
        return (self.alphas - 1.) / (np.sum(self.alphas) - self.K)

    def log_likelihood(self, x):
        return gammaln(np.sum(self.alphas))\
               - np.sum(gammaln(self.alphas))\
               + np.sum((self.alphas - 1.) * np.log(x))

    def log_partition(self):
        return - gammaln(np.sum(self.alphas))\
               + np.sum(gammaln(self.alphas))

    def entropy(self):
        return self.log_partition() - np.sum((self.alphas - 1.) * (digamma(self.alphas) - digamma(np.sum(self.alphas))))


class StickBreaking(Distribution):

    def __init__(self, K=None, gammas=None, deltas=None):
        self.K = K
        self.gammas = gammas
        self.deltas = deltas

    @property
    def params(self):
        return self.gammas, self.deltas

    @params.setter
    def params(self, values):
        self.gammas, self.deltas = values

    @property
    def dim(self):
        return self.K

    def rvs(self, size=1):
        # stick-breaking construction
        betas = npr.beta(self.gammas, self.deltas)
        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        probs = probs / probs.sum()
        # probs[np.where(probs < 1e-16)[0]] = 0.
        return probs

    def mean(self):
        # mean of stick-breaking
        betas = self.gammas / (self.gammas + self.deltas)
        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        probs = probs / probs.sum()
        # probs[np.where(probs < 1e-16)[0]] = 0.
        return probs

    def mode(self):
        # mode of stick-breaking
        betas = np.zeros((self.K, ))
        for k in range(self.K):
            if self.gamma[k] > 1. and self.delta[k] > 1.:
                betas[k] = (self.gamma[k] - 1.) / (self.gamma[k] + self.delta[k] - 2.)
            elif self.gamma[k] == 1. and self.delta[k] == 1.:
                betas[k] = 1.
            elif self.gamma[k] < 1. and self.delta[k] < 1.:
                betas[k] = 1.
            elif self.gamma[k] <= 1. and self.delta[k] > 1.:
                betas[k] = 0.
            elif self.gamma[k] > 1. and self.delta[k] <= 1.:
                betas[k] = 1.
            else:
                warnings.warn("Mode of Dirichlet process not defined")
                raise ValueError

        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        probs = probs / probs.sum()
        # probs[np.where(probs < 1e-16)[0]] = 0.
        return probs

    def log_likelihood(self, x):
        raise NotImplementedError

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
