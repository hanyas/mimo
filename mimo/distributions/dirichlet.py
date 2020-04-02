import numpy as np
import numpy.random as npr
from scipy.special import digamma, gammaln

from mimo.abstractions import Distribution

import warnings


class Dirichlet(Distribution):

    def __init__(self, K=None, alphas=None):
        self.K = K
        self.alphas = alphas

    @property
    def params(self):
        return self.K, self.alphas

    @params.setter
    def params(self, values):
        self.K, self.alphas = values

    @property
    def dim(self):
        return self.K

    def rvs(self, size=None):
        return npr.dirichlet(self.alphas, size)

    def mean(self):
        return self.alphas / np.sum(self.alphas)

    def mode(self):
        if np.all(self.alphas > 1.):
            return (self.alphas - 1.) / (np.sum(self.alphas) - self.K)
        else:
            warnings.warn("Mode of Dirichlet distribution not defined")
            return None

    def log_likelihood(self, x):
        return gammaln(np.sum(self.alphas))\
               - np.sum(gammaln(self.alphas))\
               + np.sum((self.alphas - 1.) * np.log(x))

    def log_partition(self):
        return - gammaln(np.sum(self.alphas))\
               + np.sum(gammaln(self.alphas))

    def entropy(self):
        return self.log_partition() - np.sum((self.alphas - 1.) * (digamma(self.alphas) - digamma(np.sum(self.alphas))))

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            counts = np.bincount(data, minlength=self.K)
        else:
            counts = sum(np.bincount(d, minlength=self.K) for d in data)
        return counts

    def get_weighted_statistics(self, data, weights):
        if isinstance(weights, np.ndarray):
            assert weights.ndim in (1, 2)
            if data is None or weights.ndim == 2:
                # when weights is 2D or data is None, the weights are expected
                # indicators and data is just a placeholder; nominally data
                # should be np.arange(K)[na,:].repeat(N,axis=0)
                counts = np.sum(np.atleast_2d(weights), axis=0)
            else:
                # when weights is 1D, data is indices and we do a weighted
                # bincount
                counts = np.bincount(data, weights, minlength=self.K)
        else:
            if len(weights) == 0:
                counts = np.zeros(self.K, dtype=int)
            else:
                data = data if data else [None] * len(weights)
                counts = sum(self.get_weighted_statistics(d, w) for d, w in zip(data, weights))
        return counts


class StickBreaking(Distribution):

    def __init__(self, K=None, gammas=None, deltas=None):
        self.K = K
        self.gammas = gammas
        self.deltas = deltas

    @property
    def params(self):
        return self.K, self.gammas, self.deltas

    @params.setter
    def params(self, values):
        self.K, self.gammas, self.deltas = values

    @property
    def dim(self):
        return self.K

    def rvs(self, size=None):
        # stick-breaking construction
        betas = npr.beta(self.gammas, self.deltas)
        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        # probs = probs / probs.sum()
        # probs[np.where(probs < 1e-16)[0]] = 0.
        return probs

    def mean(self):
        # mean of stick-breaking
        betas = self.gammas / (self.gammas + self.deltas)
        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        # probs = probs / probs.sum()
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
                return None

        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        # probs = probs / probs.sum()
        # probs[np.where(probs < 1e-16)[0]] = 0.
        return probs

    def log_likelihood(self, x):
        raise NotImplementedError

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            counts = np.bincount(data, minlength=self.K)
        else:
            counts = sum(np.bincount(d, minlength=self.K) for d in data)
        return counts

    def get_weighted_statistics(self, data, weights):
        if isinstance(weights, np.ndarray):
            assert weights.ndim in (1, 2)
            if data is None or weights.ndim == 2:
                # when weights is 2D or data is None, the weights are expected
                # indicators and data is just a placeholder; nominally data
                # should be np.arange(K)[na,:].repeat(N,axis=0)
                counts = np.sum(np.atleast_2d(weights), axis=0)
            else:
                # when weights is 1D, data is indices and we do a weighted
                # bincount
                counts = np.bincount(data, weights, minlength=self.K)
        else:
            if len(weights) == 0:
                counts = np.zeros(self.K, dtype=int)
            else:
                data = data if data else [None] * len(weights)
                counts = sum(self.get_weighted_statistics(d, w) for d, w in zip(data, weights))
        return counts

