import abc
import copy
from operator import add, sub
from functools import reduce

from future.utils import with_metaclass
from mimo.util.data import islist


# Base classes
class Distribution(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def rvs(self, size=1):
        # random variates (samples)
        pass

    @abc.abstractmethod
    def log_likelihood(self, x):
        """
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        """
        pass

    @abc.abstractmethod
    def mean(self):
        pass

    @abc.abstractmethod
    def mode(self):
        pass

    @abc.abstractmethod
    def log_partition(self):
        pass

    @abc.abstractmethod
    def entropy(self):
        pass


class BayesianDistribution(with_metaclass(abc.ABCMeta, Distribution)):
    @abc.abstractmethod
    def empirical_bayes(self, data):
        """
        (optional) set hyperparameters via empirical bayes
        e.g. treat argument as a pseudo-dataset for exponential family
        """
        raise NotImplementedError

    # Algorithm interfaces for inference in distributions
    @abc.abstractmethod
    def resample(self, data=[]):
        pass

    @abc.abstractmethod
    def copy_sample(self):
        """
        return an object copy suitable for making lists of posterior samples
        (override this method to prevent copying shared structures into each sample)
        """
        return copy.deepcopy(self)

    @abc.abstractmethod
    def resample_and_copy(self):
        self.resample()
        return self.copy_sample()

    @abc.abstractmethod
    def expected_log_likelihood(self, data):
        pass

    @abc.abstractmethod
    def meanfield_update(self, data, weights):
        pass

    @abc.abstractmethod
    def variational_lowerbound(self):
        raise NotImplementedError

    @abc.abstractmethod
    def meanfield_sgdstep(self, stats, weights, prob, stepsize):
        pass

    @abc.abstractmethod
    def max_likelihood(self, data, weights=None):
        """
        sets the parameters set to their maximum likelihood values given the
        (weighted) data
        """
        pass

    @property
    def nb_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def max_aposteriori(self, data, weights=None):
        """
        sets the parameters to their MAP values given the (weighted) data
        analogous to max_likelihood but includes hyperparameters
        """
        pass


class Conditional(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def rvs(self, x):
        # random variates (samples)
        pass

    @abc.abstractmethod
    def log_likelihood(self, y, x):
        """
        log likelihood (either log probability mass function or log probability
        density function) of y, which has the same type as the output of rvs()
        x is a conditional variable of the density
        """
        pass

    @abc.abstractmethod
    def mean(self, x):
        pass

    @abc.abstractmethod
    def mode(self, x):
        pass

    @abc.abstractmethod
    def log_partition(self):
        pass

    @abc.abstractmethod
    def entropy(self):
        pass


class BayesianConditional(with_metaclass(abc.ABCMeta, Conditional)):
    def empirical_bayes(self, data):
        """
        (optional) set hyperparameters via empirical bayes
        e.g. treat argument as a pseudo-dataset for exponential family
        """
        raise NotImplementedError

    @abc.abstractmethod
    def resample(self, data=[]):
        pass

    def resample_and_copy(self):
        self.resample()
        return self.copy_sample()

    @abc.abstractmethod
    def expected_log_likelihood(self, x):
        pass

    @abc.abstractmethod
    def meanfield_update(self, data, weights):
        pass

    def variational_lowerbound(self):
        raise NotImplementedError

    @abc.abstractmethod
    def meanfield_sgdstep(self, stats, weights, prob, stepsize):
        pass

    @abc.abstractmethod
    def max_likelihood(self, data, weights=None):
        """
        sets the parameters set to their maximum likelihood values given the
        (weighted) data
        """
        pass

    @property
    def nb_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def max_aposteriori(self, data, weights=None):
        """
        sets the parameters to their MAP values given the (weighted) data
        analogous to max_likelihood but includes hyperparameters
        """
        pass


class Statistics(tuple):

    def __new__(cls, x):
        return tuple.__new__(Statistics, x)

    def __add__(self, y):
        gsum = lambda x, y: reduce(lambda a, b: list(map(add, a, b)) if islist(x, y) else a + b, [x, y])
        return Statistics(tuple(map(gsum, self, y)))

    def __sub__(self, y):
        gsub = lambda x, y: reduce(lambda a, b: list(map(sub, a, b)) if islist(x, y) else a - b, [x, y])
        return Statistics(tuple(map(gsub, self, y)))

    def __mul__(self, a):
        return Statistics(a * e for e in self)

    def __rmul__(self, a):
        return Statistics(a * e for e in self)
