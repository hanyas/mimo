import numpy as np
import numpy.random as npr

from mimo.abstraction import Distribution


class Categorical(Distribution):

    def __init__(self, K=None, probs=None):
        self.K = K
        self.probs = probs

        if K is not None and probs is None:
            self.probs = 1. / self.K * np.ones((self.K, ))

    @property
    def params(self):
        return self.probs

    @params.setter
    def params(self, values):
        self.probs = values

    @property
    def nb_params(self):
        return len(self.probs) - 1

    @property
    def dim(self):
        return self.K

    def rvs(self, size=1):
        return npr.choice(a=self.K, p=self.probs, size=size)

    def mean(self):
        raise NotImplementedError

    def mode(self):
        return np.argmax(self.probs)

    def log_likelihood(self, x):
        bads = np.isnan(x)
        log_lik = np.zeros_like(x, dtype=np.double)
        err = np.seterr(invalid='ignore', divide='ignore')
        log_lik[~bads] = np.log(self.probs)[list(x[~bads])]  # log(0) can happen, no warning
        np.seterr(**err)
        return log_lik

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def statistics(self, data):
        # Stats are reduced
        if isinstance(data, np.ndarray):
            return np.bincount(data, minlength=self.K)
        else:
            return sum(list(map(self.statistics, data)))

    def weighted_statistics(self, data, weights):
        # Stats are reduced
        if isinstance(weights, np.ndarray):
            assert weights.ndim in (1, 2)
            if data is None or weights.ndim == 2:
                # when weights is 2D or data is None, the weights are expected
                # indicators and data is just a placeholder; nominally data
                # should be np.arange(K)[None, :].repeat(N,axis=0)
                return np.sum(np.atleast_2d(weights), axis=0)
            else:
                # when weights is 1D, data is indices and we do a weighted
                # bincount
                return np.bincount(data, weights, minlength=self.K)
        else:
            if len(weights) == 0:
                return np.zeros(self.K, dtype=int)
            else:
                data = data if data else [None] * len(weights)
                return sum(list(map(self.weighted_statistics, data, weights)))

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        counts = self.statistics(data) if weights is None\
            else self.weighted_statistics(data, weights)
        self.probs = counts / counts.sum()
        return self
