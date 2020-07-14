import copy

import numpy as np

from mimo.distributions import Categorical
from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import LinearGaussianWithDiagonalPrecision
from mimo.distributions import TiedGaussiansWithPrecision

from mimo.util.stats import multivariate_gaussian_loglik as mvn_logpdf
from mimo.util.stats import multivariate_studentt_loglik as mvt_logpdf


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
        self.prior.nat_param = self.likelihood.statistics(data)
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

    def log_marginal_likelihood(self):
        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition()
        return log_partition_posterior - log_partition_prior

    def posterior_predictive_gaussian(self):
        mu, kappa, psi, nu = self.posterior.params
        df = nu - self.likelihood.dim + 1
        c = 1. + 1. / kappa
        lmbda = df * psi / c
        return mu, lmbda

    def log_posterior_predictive_gaussian(self, x):
        mu, lmbda = self.posterior_predictive_gaussian()
        return GaussianWithPrecision(mu=mu, lmbda=lmbda).log_likelihood(x)

    def posterior_predictive_studentt(self):
        mu, kappa, psi, nu = self.posterior.params
        df = nu - self.likelihood.dim + 1
        c = 1. + 1. / kappa
        lmbda = df * psi / c
        return mu, lmbda, df

    def log_posterior_predictive_studentt(self, x):
        mu, lmbda, df = self.posterior_predictive_studentt()
        return mvt_logpdf(x, mu, lmbda, df)


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

    def log_marginal_likelihood(self):
        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition()
        return log_partition_posterior - log_partition_prior

    def posterior_predictive_gaussian(self):
        mu, kappas, alphas, betas = self.posterior.params
        c = 1. + 1. / kappas
        lmbdas = (alphas / betas) * 1. / c
        return mu, lmbdas

    def log_posterior_predictive_gaussian(self, x):
        mu, lmbdas = self.posterior_predictive_gaussian()
        return GaussianWithDiagonalPrecision(mu=mu, lmbdas=lmbdas).log_likelihood(x)

    def posterior_predictive_studentt(self):
        mu, kappas, alphas, betas = self.posterior.params
        dfs = 2. * alphas
        c = 1. + 1. / kappas
        lmbdas = (alphas / betas) * 1. / c
        return mu, lmbdas, dfs

    def log_posterior_predictive_studentt(self, x):
        mu, lmbdas, dfs = self.posterior_predictive_studentt()
        log_posterior = 0.
        for _x, _mu, _lmbda, _df in zip(x, mu, lmbdas, dfs):
            log_posterior += mvt_logpdf(_x.reshape(-1, 1),
                                        _mu.reshape(-1, 1),
                                        _lmbda.reshape(-1, 1, 1),
                                        _df)
        return log_posterior


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


class LinearGaussianWithMatrixNormalWishartAndAutomaticRelevance:

    def __init__(self, prior, hypprior, likelihood=None, affine=True):
        # Matrix-Normal-Wishart prior
        self.prior = prior

        # Matrix-Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # Diagonal Gamma hyper prior
        self.hypprior = hypprior

        # Diagonal Gamma hyper posterior
        self.hypposterior = copy.deepcopy(hypprior)

        # Linear Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda = self.prior.rvs()
            self.likelihood = LinearGaussianWithPrecision(A=A, lmbda=lmbda,
                                                          affine=affine)

    def empirical_bayes(self, y, x):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, y=[], x=[], nb_iter=5):
        for _ in range(nb_iter):
            stats = self.likelihood.statistics(y, x)
            self.posterior.nat_param = self.prior.nat_param + stats

            self.likelihood.params = self.posterior.rvs()
            A, lmbda = self.likelihood.params

            hyperstats = self.prior.statistics(A, lmbda)
            self.hypposterior.nat_param = self.hypprior.nat_param + hyperstats

            self.prior.matnorm.K = np.diag(self.hypposterior.rvs())
        return self


class LinearGaussianWithMatrixNormalWishart:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal Wishart prior.
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

    def empirical_bayes(self, y, x):
        self.prior.nat_param = self.likelihood.statistics(y, x)
        self.likelihood.params = self.prior.rvs()
        return self

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

    def posterior_predictive_gaussian(self, x):
        x = np.reshape(x, (-1, self.likelihood.dcol))
        nb = x.shape[0]

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.drow + 1
        mus = np.einsum('kh,...h->...k', M, x)

        c = 1. + np.einsum('...k,...kh,...h->...', x, np.linalg.inv(K), x)
        lmbdas = np.einsum('kh,...->...kh', psi, df / c)
        if nb == 1:
            return mus[0], lmbdas[0]
        else:
            return mus, lmbdas

    def log_posterior_predictive_gaussian(self, y, x):
        mus, lmbdas = self.posterior_predictive_gaussian(x)
        return mvn_logpdf(y, mus, lmbdas)

    def posterior_predictive_studentt(self, x):
        x = np.reshape(x, (-1, self.likelihood.dcol))
        nb = x.shape[0]

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.drow + 1
        mus = np.einsum('kh,...h->...k', M, x)

        c = 1. + np.einsum('...k,...kh,...h->...', x, np.linalg.inv(K), x)
        lmbdas = np.einsum('kh,...->...kh', psi, df / c)

        if nb == 1:
            return mus[0], lmbdas[0], df
        else:
            return mus, lmbdas, df

    def log_posterior_predictive_studentt(self, y, x):
        mus, lmbdas, df = self.posterior_predictive_studentt(x)
        return mvt_logpdf(y, mu=mus, lmbda=lmbdas, df=df)
