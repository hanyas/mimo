import copy

import numpy as np

from mimo.distributions import Categorical
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import TiedGaussiansWithPrecision
from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import NormalWishart

from mimo.util.matrix import invpd


class CategoricalWithDirichlet:
    """
    This class is a categorical distribution over labels
     with a Dirichlet distribution as prior.
    Parameters:
        probs, a vector encoding a finite pmf
    """

    def __init__(self, prior,  likelihood=None):
        # Dirichlet prior
        self.prior = prior

        # Dirichlet posterior
        self.posterior = copy.deepcopy(prior)

        # Categorical likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            probs = self.prior.rvs()
            self.likelihood = Categorical(K=len(probs), probs=probs)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mean()  # mode might be undefined
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        stats = self.likelihood.statistics(data)
        self.posterior.nat_param = self.prior.nat_param + stats

        _probs = self.posterior.rvs()
        self.likelihood.params = np.clip(_probs, np.spacing(1.), np.inf)
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy


class CategoricalWithStickBreaking:
    """
    This class is a categorical distribution over labels
     with a stick-breaking process as prior.
    Parameters:
        probs, a vector encoding a finite pmf
    """

    def __init__(self, prior, likelihood=None):
        # stick-breaking prior
        self.prior = prior

        # stick-breaking posterior
        self.posterior = copy.deepcopy(prior)

        # Categorical likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            probs = self.prior.rvs()
            self.likelihood = Categorical(K=len(probs), probs=probs)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.likelihood.params = self.posterior.mean()  # mode might not exist
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        counts = self.likelihood.statistics(data)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.likelihood.params = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = (1. - stepsize) * self.posterior.gammas\
                                + stepsize * (self.prior.gammas + 1. / prob * counts)
        self.posterior.deltas = (1. - stepsize) * self.posterior.deltas\
                                + stepsize * (self.prior.deltas + 1. / prob * cumcounts)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy


class GaussianWithNormalWishart:
    """
    Multivariate Gaussian distribution class.
    Uses a Normal-Wishart prior and posterior
    Parameters are mean and precision matrix:
        mu, lmbda
    """

    def __init__(self, prior, likelihood=None):
        # Normal-Wishart conjugate
        self.prior = prior

        # Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mu, lmbda = self.prior.rvs()
            self.likelihood = GaussianWithPrecision(mu=mu, lmbda=lmbda)

    def empirical_bayes(self, data):
        self.prior.nat_param = self.likelihood.get_statistics(data)
        self.likelihood.params = self.prior.rvs()
        return self

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()  # mode of wishart might not exist
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        stats = self.likelihood.statistics(data)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy

    def log_marginal_likelihood(self, x):
        x = np.atleast_2d(x)

        stats = self.likelihood.get_statistics(x)
        natparam = self.prior.nat_param + stats
        params = NormalWishart.nat_to_std(natparam)

        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition(params)

        return log_partition_posterior - log_partition_prior\
               - 0.5 * len(x) * self.likelihood.dim * np.log(np.pi)

    def log_posterior_predictive_gaussian(self, x):
        x = np.atleast_2d(x)

        stats = self.likelihood.get_statistics(x)
        natparam = self.posterior.nat_param + stats
        mu, kappa, psi, nu = NormalWishart.nat_to_std(natparam)

        loc = mu
        scale = invpd(psi * kappa)

        from mimo.util.stats import multivariate_gaussian_loglik
        return multivariate_gaussian_loglik(x, loc, scale)

    def log_posterior_predictive_studentt(self, x):
        x = np.atleast_2d(x)

        stats = self.likelihood.get_statistics(x)
        natparam = self.posterior.nat_param + stats
        mu, kappa, psi, nu = NormalWishart.nat_to_std(natparam)

        # Following Bishop notation
        loc = mu
        df = nu + 1 - self.likelihood.dim
        scale = invpd(df * kappa * psi / (1 + kappa))

        from mimo.util.stats import multivariate_studentt_loglik
        return multivariate_studentt_loglik(x, loc, scale, df)


class GaussianWithNormalGamma:
    """
    Multivariate Diagonal Gaussian distribution class.
    Uses a Normal-Gamma prior and posterior
    Parameters are mean and precision matrix:
        mu, lmbdas
    """

    def __init__(self, prior, likelihood=None):
        # Normal-Gamma conjugate
        self.prior = prior

        # Normal-Gamma posterior
        self.posterior = copy.deepcopy(prior)

        # Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mu, lmbdas = self.prior.rvs()
            self.likelihood = GaussianWithDiagonalPrecision(mu=mu, lmbdas=lmbdas)

    def empirical_bayes(self, data):
        self.prior.nat_param = self.likelihood.statistics(data)
        self.likelihood.params = self.prior.rvs()
        return self

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()  # mode of gamma might not exist
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        stats = self.likelihood.statistics(data)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.prior.cross_entropy(self.posterior)
        return q_entropy - qp_cross_entropy


class TiedGaussiansWithNormalWishart:

    def __init__(self, prior, likelihood=None):
        # Tied Normal Wishart prior
        self.prior = prior

        # Tied Normal Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # TiedGaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mus, lmbda = self.prior.rvs()
            self.likelihood = TiedGaussiansWithPrecision(mus=mus, lmbda=lmbda)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights):
        stats = self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    # Gibbs sampling
    def resample(self, data=[], labels=[]):
        stats = self.likelihood.statistics(data, labels)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, data, weights):
        stats = self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy


class LinearGaussianWithMatrixNormalWishart:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal/Inverse-Wishart prior.
    Parameters are linear transf. and covariance matrix:
        A, lmbda
    """

    def __init__(self, prior, likelihood=None, affine=True):
        # Matrix-Normal-Wishart prior
        self.prior = prior

        # Matrix-Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # Linear Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda = self.prior.rvs()
            self.likelihood = LinearGaussianWithPrecision(A=A, lmbda=lmbda,
                                                          affine=affine)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, y, x, weights=None):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, y=[], x=[]):
        stats = self.likelihood.statistics(y, x)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, y, x, weights=None):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, y, x, weights, prob, stepsize):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy

    def predictive_posterior_gaussian(self, x):
        if self.likelihood.affine:
            x = np.hstack((x, 1.))

        M, V, psi, nu = self.posterior.params

        # https://tminka.github.io/papers/minka-gaussian.pdf
        mu = M @ x

        # variance of approximate Gaussian
        sigma = psi / nu  # Misleading in Minka

        return mu, sigma, nu

    def predictive_posterior_studentt(self, x):
        if self.likelihood.affine:
            x = np.hstack((x, 1.))

        xxT = np.outer(x, x)

        M, V, psi, nu = self.posterior.params

        # https://tminka.github.io/papers/minka-linear.pdf
        c = 1. - x.T @ np.linalg.inv(np.linalg.inv(V) + xxT) @ x

        # https://tminka.github.io/papers/minka-gaussian.pdf
        df = nu
        mu = M @ x

        # variance of a student-t
        sigma = (1. / c) * psi / df  # Misleading in Minka
        var = sigma * df / (df - 2)

        return mu, sigma, nu
