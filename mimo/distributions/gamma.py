import numpy as np
import numpy.random as npr

from scipy.special import gammaln, digamma
from mimo.utils.abstraction import Statistics as Stats


class Gamma:
    # In comparison to a Wishart distribution
    # alpha = nu / 2.
    # beta = 1. / (2. * psi)

    def __init__(self, dim, alphas, betas):
        self.dim = dim

        self.alphas = alphas  # shape
        self.betas = betas  # rate

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        alphas = params[0] - 1
        betas = - params[1]
        return Stats([alphas, betas])

    @staticmethod
    def nat_to_std(natparam):
        alphas = natparam[0] + 1
        betas = - natparam[1]
        return alphas, betas

    def mean(self):
        return self.alphas / self.betas

    def mode(self):
        assert np.all(self.alphas >= 1.)
        return (self.alphas - 1.) / self.betas

    def rvs(self, size=1):
        # numpy uses a different parameterization
        return npr.gamma(self.alphas, 1. / self.betas)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            logx = np.log(data)
            x = data
            n = np.ones((data.shape[0], ))

            return Stats([logx, n, x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            logx = np.einsum('n,nk->nk', weights, np.log(data))
            x = np.einsum('n,nk->nk', weights, data)
            n = weights

            return Stats([logx, n, x, n])
        else:
            return list(map(self.weighted_statistics, data, weights))

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def log_likelihood(self, x):
        # not vectorized
        log_lik = np.sum((self.alphas - 1.) * np.log(x) - self.betas * x)
        return - self.log_partition() + self.log_base() + log_lik

    def expected_statistics(self):
        E_log_x = digamma(self.alphas) - np.log(self.betas)
        E_x = self.alphas / self.betas
        return E_log_x, E_x

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))


class InverseGamma:

    def __init__(self, dim, alphas, betas):
        self.dim = dim

        self.alphas = alphas  # shape
        self.betas = betas  # rate

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        alphas = - params[0] - 1
        betas = - params[1]
        return Stats([alphas, betas])

    @staticmethod
    def nat_to_std(natparam):
        alphas = - natparam[0] - 1
        betas = - natparam[1]
        return alphas, betas

    def mean(self):
        assert np.all(self.alphas >= 1.)
        return self.betas / (self.alphas - 1)

    def mode(self):
        return self.betas / (self.alphas + 1.)

    def rvs(self, size=1):
        # numpy uses a different parameterization
        return 1. / npr.gamma(self.alphas, 1. / self.betas)

    def statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]

            logx = np.log(data)
            x = 1. / data
            n = np.ones((data.shape[0], ))

            return Stats([logx, n, x, n])
        else:
            return list(map(self.statistics, data))

    def weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(axis=1)
            data = data[idx]
            weights = weights[idx]

            logx = np.einsum('n,nk->nk', weights, np.log(data))
            x = np.einsum('n,nk->nk', weights, 1. / data)
            n = weights

            return Stats([logx, n, x, n])
        else:
            return list(map(self.weighted_statistics, data, weights))

    @property
    def base(self):
        return 1.

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def log_likelihood(self, x):
        # not vectorized
        log_lik = np.sum((- self.alphas - 1.) * np.log(x) - self.betas / x)
        return - self.log_partition() + self.log_base() + log_lik

    def expected_statistics(self):
        E_log_x = np.log(self.betas) - digamma(self.alphas)
        E_x_inv = self.alphas / self.betas
        return E_log_x, E_x_inv

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (nat_param[0].dot(stats[0]) + np.dot(nat_param[1], stats[1]))
