import numpy as np
import numpy.random as npr

from mimo.abstractions import Distribution


class Categorical(Distribution):

    def __init__(self,  K=None, probs=None):
        self.K = K
        self.probs = probs

    @property
    def params(self):
        return self.K, self.probs

    @params.setter
    def params(self, values):
        self.K, self.probs = values

    @property
    def num_parameters(self):
        return len(self.probs) - 1

    @property
    def dim(self):
        return self.K

    def rvs(self, size=None):
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
