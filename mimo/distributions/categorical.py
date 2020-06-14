import numpy as np
import numpy.random as npr

from mimo.abstraction import Distribution


class Categorical(Distribution):

    def __init__(self, K=None, probs=None):
        self.K = K
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
        out = np.zeros_like(x, dtype=np.double)
        bads = np.isnan(x)
        err = np.seterr(divide='ignore')
        out[~bads] = np.log(self.probs)[list(x[~bads])]  # log(0) can happen, no warning
        np.seterr(**err)
        return out

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

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        counts = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.probs = counts / counts.sum()
        return self
