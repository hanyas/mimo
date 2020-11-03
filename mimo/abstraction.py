import abc
from operator import add, sub
from functools import reduce

from future.utils import with_metaclass
from mimo.util.data import islist


# Base classes
class Distribution(with_metaclass(abc.ABCMeta)):

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    def rvs(self, size=1):
        pass

    def mean(self):
        pass

    def mode(self):
        pass

    def log_likelihood(self, x):
        pass

    def log_partition(self):
        pass

    def entropy(self):
        pass

    def max_likelihood(self, data, weights=None):
        pass


class BayesianDistribution(with_metaclass(abc.ABCMeta, Distribution)):

    def empirical_bayes(self, data):
        raise NotImplementedError

    def max_aposteriori(self, data, weights=None):
        pass

    def resample(self, data=[]):
        pass

    def meanfield_update(self, data, weights):
        pass

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        pass

    def variational_lowerbound(self):
        raise NotImplementedError

    def expected_log_likelihood(self, data):
        pass


class Conditional(with_metaclass(abc.ABCMeta)):

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    def rvs(self, x, size=1):
        pass

    def log_likelihood(self, y, x):
        pass

    def mean(self, x):
        pass

    def mode(self, x):
        pass

    def log_partition(self, x):
        pass

    def entropy(self, x):
        pass

    def max_likelihood(self, y, x, weights=None):
        pass


class BayesianConditional(with_metaclass(abc.ABCMeta, Conditional)):

    def empirical_bayes(self, y, x):
        raise NotImplementedError

    def max_aposteriori(self, y, x, weights=None):
        pass

    def resample(self, y=[], x=[]):
        pass

    def meanfield_update(self, y, x, weights):
        pass

    def meanfield_sgdstep(self, y, x, weights, prob, stepsize):
        pass

    def variational_lowerbound(self, x):
        pass

    def expected_log_likelihood(self, y, x):
        pass


class MixtureDistribution(with_metaclass(abc.ABCMeta)):

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    def rvs(self, size=1):
        pass

    def log_likelihood(self, obs):
        pass

    def log_scores(self, obs):
        pass

    def scores(self, obs):
        pass

    def max_likelihood(self, obs, weights=None):
        pass


class BayesianMixtureDistribution(with_metaclass(abc.ABCMeta)):

    @property
    def used_labels(self):
        raise NotImplementedError

    def add_data(self, obs, **kwargs):
        pass

    def clear_data(self):
        pass

    def has_data(self):
        pass

    def max_aposterior(self, obs, **kwargs):
        pass

    def resample(self, obs=[], labels=[], **kwargs):
        pass

    def expected_scores(self, obs):
        pass

    def meanfield_update(self, obs):
        pass

    def meanfield_coordinate_descent(self, **kwargs):
        pass

    def meanfield_sgdstep(self, obs, prob, stepsize):
        pass

    def meanfield_stochastic_descent(self, **kwargs):
        pass

    def variational_lowerbound(self, obs, weights):
        pass


class ConditionalMixtureDistribution(with_metaclass(abc.ABCMeta)):

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    def rvs(self, size=1):
        pass

    def log_likelihood(self, y, x):
        pass

    def log_scores(self, y, x):
        pass

    def scores(self, y, x):
        pass

    def max_likelihood(self, y, x, weights=None):
        pass


class BayesianConditionalMixtureDistribution(with_metaclass(abc.ABCMeta)):

    @property
    def used_labels(self):
        raise NotImplementedError

    def add_data(self, obs, **kwargs):
        pass

    def clear_data(self):
        pass

    def has_data(self):
        pass

    def max_aposterior(self, y, x, **kwargs):
        pass

    def resample(self, y=[], x=[], z=[], **kwargs):
        pass

    def expected_scores(self, y, x):
        pass

    def meanfield_update(self, y, x):
        pass

    def meanfield_coordinate_descent(self, **kwargs):
        pass

    def meanfield_sgdstep(self, y, x, prob, stepsize):
        pass

    def meanfield_stochastic_descent(self, **kwargs):
        pass

    def variational_lowerbound(self, y, x, weights):
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
