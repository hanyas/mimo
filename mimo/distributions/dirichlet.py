import numpy as np
import numpy.random as npr

from scipy.special import digamma, gammaln, betaln

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
        log_lik = np.sum((self.alphas - 1.) * np.log(x))
        return - self.log_partition() + self.log_base() + log_lik

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            logx = np.log(data)

            return logx
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            logx = np.einsum('n,nk->nk', weights, np.log(data))

            return logx
        else:
            return list(map(self.weighted_statistics, data, weights))

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        alphas = params - 1.
        return alphas

    @staticmethod
    def nat_to_std(natparam):
        alphas = natparam + 1.
        return alphas

    def log_partition(self):
        return np.sum(gammaln(self.alphas)) - gammaln(np.sum(self.alphas))

    def expected_statistics(self):
        E_log_x = digamma(self.alphas) - digamma(np.sum(self.alphas))
        return E_log_x

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - nat_param.dot(stats)

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - nat_param.dot(stats)

    def expected_log_likelihood(self):
        return self.expected_statistics()


class TruncatedStickBreaking(Distribution):
    # This stick-breaking construction is meant to
    # be used as a truncated prior/posterior as suggest in
    # "Gibbs sampling methods for stick-breaking priors"
    # by Ishwaran and James, 2001
    # and
    # "Variational Inference for Dirichlet Process Mixtures"
    # by Blei and Jordan, 2006

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

    def rvs(self, size=1, truncate=True):
        # stick-breaking construction
        betas = npr.beta(self.gammas[:-1], self.deltas[:-1])
        betas = np.hstack((betas, 1.))

        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        return probs

    def mean(self):
        # mean of stick-breaking
        betas = self.gammas[:-1] / (self.gammas[:-1] + self.deltas[:-1])
        betas = np.hstack((betas, 1.))

        probs = np.zeros((self.K, ))
        probs[0] = betas[0]
        probs[1:] = betas[1:] * np.cumprod(1.0 - betas[:-1])

        return probs

    def mode(self):
        # mode of stick-breaking
        betas = np.zeros((self.K, ))
        betas[-1] = 1.
        for k in range(self.K - 1):
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

        return probs

    def log_likelihood(self, x):
        raise NotImplementedError

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        gammas = params[0] - 1.
        deltas = params[1] - 1.
        return gammas, deltas

    @staticmethod
    def nat_to_std(natparam):
        gammas = natparam[0] + 1.
        deltas = natparam[1] + 1.
        return gammas, deltas

    def log_partition(self):
        return np.sum(betaln(self.gammas, self.deltas))

    def expected_statistics(self):
        E_log_stick = digamma(self.gammas) - digamma(self.gammas + self.deltas)
        E_log_rest = digamma(self.deltas) - digamma(self.gammas + self.deltas)
        return E_log_stick, E_log_rest

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + nat_param[1].dot(stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + nat_param[1].dot(stats[1]))

    def expected_log_likelihood(self):
        return self.expected_statistics()
