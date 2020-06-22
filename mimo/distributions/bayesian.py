import copy
from abc import ABC

import numpy as np

from mimo.abstraction import Statistics as Stats

from mimo.distributions import Categorical
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import TiedGaussians
from mimo.distributions import LinearGaussian
from mimo.distributions import NormalWishart

from mimo.util.matrix import blockarray, inv_pd


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
        scale = inv_pd(psi * kappa)

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
        scale = inv_pd(df * kappa * psi / (1 + kappa))

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


class TiedGaussiansWithNormalInverseWishart(TiedGaussians, ABC):

    def __init__(self, prior, mus=None, sigma=None):
        # Tied Normal Inverse Wishart Prior
        self.prior = prior

        # Tied Normal Inverse Wishart Prior posterior
        self.posterior = copy.deepcopy(prior)

        if mus is None or sigma is None:
            mus, sigma = self.prior.rvs()

        super(TiedGaussiansWithNormalInverseWishart, self).__init__(mus=mus, sigma=sigma)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights):
        stats = []
        for k, c in enumerate(self.components):
            _weights = None if weights is None else [_w[:, k] for _w in weights]
            stats.append(c.statistics(data) if _weights is None
                         else c.weighted_statistics(data, _weights))
        self.posterior.nat_param = self.prior.nat_param + Stats(stats)

        self.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, data=[], labels=[]):
        stats = []
        for k, c in enumerate(self.components):
            _data = [_d[_l == k, :] for _d, _l in zip(data, labels)]
            stats.append(c.statistics(_data))
        self.posterior.nat_param = self.prior.nat_param + Stats(stats)

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mus = self.mus.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = []
        for k, c in enumerate(self.components):
            _weights = None if weights is None else [_w[:, k] for _w in weights]
            stats.append(c.statistics(data) if _weights is None
                         else c.weighted_statistics(data, _weights))
        self.posterior.nat_param = self.prior.nat_param + Stats(stats)

        self.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = []
        for k, c in enumerate(self.components):
            _weights = None if weights is None else [_w[:, k] for _w in weights]
            stats.append(c.statistics(data) if _weights is None
                         else c.weighted_statistics(data, _weights))
        self.posterior.nat_param = self.prior.nat_param + Stats(stats)

        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        return sum([c.variational_lowerbound() for c in self.posterior.components])

    def expected_log_likelihood(self, x):
        return np.hstack()


class LinearGaussianWithMatrixNormalInverseWishart(LinearGaussian, ABC):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal/Inverse-Wishart prior.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """

    def __init__(self, prior, A=None, sigma=None):
        super(LinearGaussianWithMatrixNormalInverseWishart, self).__init__(A=A, sigma=sigma, affine=prior.affine)

        # Matrix-Normal-Inv-Wishart prior
        self.prior = prior

        # Matrix-Normal-Inv-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        self.A, self.sigma = A, sigma
        if A is None or sigma is None:
            self.A, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, y, x, weights=None):
        stats = self.posterior.statistics(y, x) if weights is None\
            else self.posterior.weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, y=[], x=[]):
        self.posterior.nat_param = self.prior.nat_param\
                                   + self.get_statistics(y, x)

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.A = self.A.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfield_update(self, y, x, weights=None):
        stats = self.get_statistics(y, x) if weights is None\
            else self.get_weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.A, self.sigma = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, y, x, weights, prob, stepsize):
        stats = self.get_statistics(y, x) if weights is None\
            else self.get_weighted_statistics(y, x, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv\
            = self.posterior.get_expected_statistics()
        a, b, c, d = self.prior.nat_param - self.posterior.nat_param

        aux = - 0.5 * np.trace(c.dot(E_Sigmainv)) + np.trace(a.T.dot(E_Sigmainv_A))\
              - 0.5 * np.trace(b.dot(E_AT_Sigmainv_A)) + 0.5 * d * E_logdetSigmainv

        logpart_diff = self.prior.log_partition() - self.posterior.log_partition()
        return aux - logpart_diff

    def expected_log_likelihood(self, y, x):
        drow = self.drow

        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv\
            = self.posterior.get_expected_statistics()

        if self.affine:
            E_Sigmainv_A, E_Sigmainv_b = E_Sigmainv_A[:, :-1], E_Sigmainv_A[:, -1]
            E_AT_Sigmainv_A, E_AT_Sigmainv_b, E_bT_Sigmainv_b =\
                E_AT_Sigmainv_A[:-1, :-1], E_AT_Sigmainv_A[:-1, -1], E_AT_Sigmainv_A[-1, -1]

        parammat = -1. / 2 * blockarray([[E_AT_Sigmainv_A, -E_Sigmainv_A.T],
                                         [-E_Sigmainv_A, E_Sigmainv]])

        xy = np.hstack((x, y))

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum('ni,ni->n', xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-drow, :-drow]), x)
            out += np.einsum(contract, y.dot(parammat[-drow:, -drow:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-drow, -drow:]), y)

        out += - drow / 2. * np.log(2 * np.pi) + 1. / 2 * E_logdetSigmainv

        if self.affine:
            out += y.dot(E_Sigmainv_b)
            out -= x.dot(E_AT_Sigmainv_b)
            out -= 1. / 2 * E_bT_Sigmainv_b

        return out

    def predictive_posterior_gaussian(self, x):
        if self.affine:
            x = np.hstack((x, 1.))

        M, V, psi, nu = self.posterior.params

        # https://tminka.github.io/papers/minka-gaussian.pdf
        mu = M @ x

        # variance of approximate Gaussian
        sigma = psi / nu  # Misleading in Minka

        return mu, sigma, nu

    def predictive_posterior_studentt(self, x):
        if self.affine:
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