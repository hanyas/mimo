import copy
from abc import ABC

import numpy as np

from mimo.distributions import Categorical

from mimo.distributions import GaussianWithPrecision
from mimo.distributions import StackedGaussiansWithPrecision
from mimo.distributions import TiedGaussiansWithPrecision

from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import StackedGaussiansWithDiagonalPrecision
from mimo.distributions import TiedGaussiansWithDiagonalPrecision

from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import StackedLinearGaussiansWithPrecision
from mimo.distributions import TiedLinearGaussiansWithPrecision

from mimo.distributions import LinearGaussianWithDiagonalPrecision
from mimo.distributions import StackedLinearGaussiansWithDiagonalPrecision
from mimo.distributions import TiedLinearGaussiansWithDiagonalPrecision

from mimo.distributions import AffineLinearGaussianWithPrecision
from mimo.distributions import StackedAffineLinearGaussiansWithPrecision

from mimo.utils.stats import multivariate_gaussian_loglik as mvn_logpdf
from mimo.utils.stats import multivariate_studentt_loglik as mvt_logpdf

from mimo.utils.stats import stacked_multivariate_gaussian_loglik as stacked_mvn_logpdf
from mimo.utils.stats import stacked_multivariate_studentt_loglik as stacked_mvt_logpdf

from mimo.utils.abstraction import Statistics as Stats


class CategoricalWithDirichlet:
    """
    Categorical distribution over labels
    with a Dirichlet distribution as prior.
    """

    def __init__(self, dim, prior, likelihood=None):
        self.dim = dim

        # Dirichlet prior
        self.prior = prior

        # Dirichlet posterior
        self.posterior = copy.deepcopy(prior)

        # Categorical likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            probs = self.prior.rvs()
            self.likelihood = Categorical(dim=self.dim, probs=probs)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, data):
        stats = self.likelihood.statistics(data)
        self.posterior.nat_param = self.prior.nat_param + stats

        probs = self.posterior.rvs()
        self.likelihood.params = np.clip(probs, np.spacing(1.), np.inf)

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, data, weights, scale, step_size):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                   + step_size * (self.prior.nat_param + 1. / scale * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        entropy = self.posterior.entropy()
        cross_entropy = self.posterior.cross_entropy(self.prior)
        return entropy - cross_entropy

    def expected_log_likelihood(self):
        return self.posterior.expected_statistics()


class CategoricalWithStickBreaking:
    """
    Categorical distribution over labels
    with a stick-breaking process as prior.
    """

    def __init__(self, dim, prior, likelihood=None):
        self.dim = dim

        # truncated stick-breaking prior
        self.prior = prior

        # truncated stick-breaking posterior
        self.posterior = copy.deepcopy(prior)

        # Categorical likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            probs = self.prior.rvs()
            self.likelihood = Categorical(dim=self.dim, probs=probs)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        acc_counts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + acc_counts

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, data):
        counts = self.likelihood.statistics(data)
        # Blei et. al Variational Inference for Dirichlet Process Mixtures
        acc_counts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + acc_counts

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, data, weights=None):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        acc_counts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + acc_counts

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, data, weights, scale, step_size):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        acc_counts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = (1. - step_size) * self.posterior.gammas\
                                + step_size * (self.prior.gammas + 1. / scale * counts)
        self.posterior.deltas = (1. - step_size) * self.posterior.deltas\
                                + step_size * (self.prior.deltas + 1. / scale * acc_counts)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        entropy = self.posterior.entropy()
        cross_entropy = self.posterior.cross_entropy(self.prior)
        return entropy - cross_entropy

    def expected_log_likelihood(self):
        return self.posterior.expected_statistics()


class GaussianWithNormalWishart:

    """
    Multivariate Gaussian distributions class.
    Uses a Normal-Wishart prior and posterior
    """

    def __init__(self, dim, prior, likelihood=None):
        self.dim = dim

        # Normal-Wishart conjugate
        self.prior = prior

        # Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # stacked Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mu, lmbda = self.prior.rvs()
            self.likelihood = GaussianWithPrecision(dim=dim, mu=mu, lmbda=lmbda)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, data, labels=None):
        stats = self.likelihood.statistics(data) if labels is None\
            else self.likelihood.weighted_statistics(data, labels)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, data, weights, scale, step_size):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                   + step_size * (self.prior.nat_param + 1. / scale * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        entropy = self.posterior.entropy()
        cross_entropy = self.posterior.cross_entropy(self.prior)
        return entropy - cross_entropy

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat * stats
        stats = self.likelihood.statistics(x, fold=False)
        log_base = self.likelihood.log_base()

        return log_base\
               + np.einsum('d,nd->n', nat_param[0], stats[0])\
               + nat_param[1] * stats[1]\
               + np.einsum('dl,ndl->n', nat_param[2], stats[2])\
               + nat_param[3] * stats[3]

    def log_marginal_likelihood(self):
        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition()
        return log_partition_posterior - log_partition_prior


class StackedGaussiansWithNormalWisharts(GaussianWithNormalWishart, ABC):
    """
    Multivariate Gaussian distributions class.
    Uses a Normal-Wishart prior and posterior
    """

    def __init__(self, size, dim, prior, likelihood=None):

        self.size = size

        if likelihood is None:
            # stacked Gaussian likelihood
            mus, lmbdas = prior.rvs()
            likelihood = StackedGaussiansWithPrecision(size=size, dim=dim,
                                                       mus=mus, lmbdas=lmbdas)

        super(StackedGaussiansWithNormalWisharts, self).__init__(dim, prior, likelihood)

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat * stats
        stats = self.likelihood.statistics(x, fold=False)
        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kd,knd->kn', nat_param[0], stats[0])\
               + np.einsum('k,kn->kn', nat_param[1], stats[1]) \
               + np.einsum('kdl,kndl->kn', nat_param[2], stats[2])\
               + np.einsum('k,kn->kn', nat_param[3], stats[3])

    def posterior_predictive_gaussian(self):
        mus, kappas, psis, nus = self.posterior.params
        dfs = nus - self.dim + 1
        cs = 1. + 1. / kappas
        lmbdas = np.einsum('k,kdl->kdl', dfs / cs, psis)
        return mus, lmbdas

    def log_posterior_predictive_gaussian(self, x):
        mus, lmbdas = self.posterior_predictive_gaussian()
        return stacked_mvn_logpdf(x, mus, lmbdas)

    def posterior_predictive_studentt(self):
        mus, kappas, psis, nus = self.posterior.params
        dfs = nus - self.dim + 1
        cs = 1. + 1. / kappas
        lmbdas = np.einsum('k,kdl->kdl', dfs / cs, psis)
        return mus, lmbdas, dfs

    def log_posterior_predictive_studentt(self, x):
        mus, lmbdas, dfs = self.posterior_predictive_studentt()
        return stacked_mvt_logpdf(x, mus, lmbdas, dfs)


class TiedGaussiansWithNormalWisharts(StackedGaussiansWithNormalWisharts, ABC):
    """
    Multivariate Gaussian distributions with tied covariance
    Uses a tied Normal-Wishart prior and posterior
    """

    def __init__(self, size, dim, prior, likelihood=None):

        if likelihood is None:
            # tied Gaussian likelihood
            mus, lmbdas = prior.rvs()
            likelihood = TiedGaussiansWithPrecision(size=size, dim=dim,
                                                    mus=mus, lmbdas=lmbdas)

        super(TiedGaussiansWithNormalWisharts, self).__init__(size, dim, prior, likelihood)


class GaussianWithNormalGamma:
    """
    Multivariate diagonal Gaussian distribution class.
    Uses a Normal-Gamma prior and posterior
    """

    def __init__(self, dim, prior, likelihood=None):
        self.dim = dim

        # Normal-Gamma conjugate
        self.prior = prior

        # Normal-Gamma posterior
        self.posterior = copy.deepcopy(prior)

        # Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            mu, lmbda_diag = self.prior.rvs()
            self.likelihood = GaussianWithDiagonalPrecision(dim=dim, mu=mu,
                                                            lmbda_diag=lmbda_diag)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, data, labels=None):
        stats = self.likelihood.statistics(data) if labels is None\
            else self.likelihood.weighted_statistics(data, labels)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, data, weights, scale, step_size):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                   + step_size * (self.prior.nat_param + 1. / scale * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        entropy = self.posterior.entropy()
        cross_entropy = self.posterior.cross_entropy(self.prior)
        return entropy - cross_entropy

    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat * stats
        stats = self.likelihood.statistics(x, fold=False)
        log_base = self.likelihood.log_base()

        return log_base\
               + np.einsum('d,nd->n', nat_param[0], stats[0])\
               + np.einsum('d,nd->n', nat_param[1], stats[1])\
               + np.einsum('d,nd->n', nat_param[2], stats[2])\
               + np.einsum('d,nd->n', nat_param[3], stats[3])

    def log_marginal_likelihood(self):
        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition()
        return log_partition_posterior - log_partition_prior


class StackedGaussiansWithNormalGammas(GaussianWithNormalGamma, ABC):
    """
    Multivariate diagonal Gaussian distributions class.
    Uses a Normal-Gamma prior and posterior
    """

    def __init__(self, size, dim, prior, likelihood=None):

        self.size = size

        if likelihood is None:
            # stacked Gaussian likelihood
            mus, lmbdas_diags = prior.rvs()
            likelihood = StackedGaussiansWithDiagonalPrecision(size=size, dim=dim,
                                                               mus=mus, lmbdas_diags=lmbdas_diags)

        super(StackedGaussiansWithNormalGammas, self).__init__(dim, prior, likelihood)

    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat * stats
        stats = self.likelihood.statistics(x, fold=False)
        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kd,knd->kn', nat_param[0], stats[0])\
               + np.einsum('kd,knd->kn', nat_param[1], stats[1])\
               + np.einsum('kd,knd->kn', nat_param[2], stats[2])\
               + np.einsum('kd,knd->kn', nat_param[3], stats[3])

    def posterior_predictive_gaussian(self):
        mus, kappas, alphas, betas = self.posterior.params
        c = 1. + 1. / kappas
        lmbda_diags = (alphas / betas) * 1. / c
        return mus, lmbda_diags

    def log_posterior_predictive_gaussian(self, x):
        mus, lmbda_diags = self.posterior_predictive_gaussian()
        lmbdas = np.eye(self.dim) * lmbda_diags[:, np.newaxis]
        return stacked_mvn_logpdf(x, mus, lmbdas)

    def posterior_predictive_studentt(self):
        mus, kappas, alphas, betas = self.posterior.params
        dfs = 2. * alphas
        c = 1. + 1. / kappas
        lmbda_diags = (alphas / betas) * 1. / c
        return mus, lmbda_diags, dfs

    def log_posterior_predictive_studentt(self, x):
        mus, lmbda_diags, dfs = self.posterior_predictive_studentt()
        lmbdas = np.eye(self.dim) * lmbda_diags[:, np.newaxis]
        return stacked_mvt_logpdf(x, mus, lmbdas, dfs)


class TiedGaussiansWithNormalGammas(StackedGaussiansWithNormalGammas, ABC):
    """
    Multivariate diagonal Gaussian distributions with tied covariance
    Uses a tied Normal-Gamma prior and posterior
    """

    def __init__(self, size, dim, prior, likelihood=None):

        if likelihood is None:
            # tied Gaussian likelihood
            mus, lmbdas_diags = prior.rvs()
            likelihood = TiedGaussiansWithDiagonalPrecision(size=size, dim=dim,
                                                            mus=mus, lmbdas_diags=lmbdas_diags)

        super(TiedGaussiansWithNormalGammas, self).__init__(size, dim, prior, likelihood)


class GaussianWithHierarchicalNormalWishart:

    def __init__(self, dim, hyper_prior, prior):
        self.dim = dim

        # init hierarchical prior
        tau, lmbda = hyper_prior.rvs()
        prior.mu, prior.lmbda = tau, lmbda

        self.hyper_prior = hyper_prior
        self.hyper_posterior = copy.deepcopy(self.hyper_prior)

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

        mu = self.prior.rvs()
        self.likelihood = GaussianWithPrecision(dim=dim, mu=mu,
                                                lmbda=lmbda)

    def empirical_bayes(self, data):
        raise NotImplementedError

    def resample(self, data, nb_iter=1):

        lmbda, mu = None, None
        for _ in range(nb_iter):
            tau, lmbda = self.hyper_posterior.rvs()

            # sampling posterior
            x, n, xxT, n = self.likelihood.statistics(data)
            self.posterior.kappa = self.prior.kappa + n
            self.posterior.mu = (self.prior.kappa * self.hyper_prior.gaussian.mu + x) / (self.prior.kappa + n)
            self.posterior.lmbda = lmbda

            mu = self.posterior.rvs()

            rho = (self.prior.kappa * mu + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu)\
                  / (self.prior.kappa + self.hyper_prior.kappa)
            kappa = self.prior.kappa + self.hyper_prior.kappa
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + self.hyper_prior.kappa * self.prior.kappa / (self.hyper_prior.kappa + self.prior.kappa)
                                * np.einsum('d,l->dl', self.hyper_prior.gaussian.mu - mu, self.hyper_prior.gaussian.mu - mu)
                                + xxT - np.einsum('d,l->dl', mu, x) - np.einsum('d,l->dl', x, mu)
                                + n * np.einsum('d,l->dl', mu, mu))
            nu = self.hyper_prior.wishart.nu + n + 1

            self.hyper_posterior.params = rho, kappa, psi, nu

        self.likelihood.params = mu, lmbda

    # Mean field
    def meanfield_update(self, data, nb_iter=25):

        vlb = []
        for i in range(nb_iter):
            # variational e-step
            x, n, xxT, n = self.likelihood.statistics(data)
            self.posterior.kappa = self.prior.kappa + n
            self.posterior.mu = (self.prior.kappa * self.hyper_posterior.gaussian.mu + x) / (self.prior.kappa + n)
            self.posterior.lmbda = self.hyper_posterior.wishart.mean()

            # variational m-step
            rho = (self.prior.kappa * self.posterior.mu + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu)\
                  / (self.prior.kappa + self.hyper_prior.kappa)
            kappa = self.prior.kappa + self.hyper_prior.kappa
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + self.hyper_prior.kappa * self.prior.kappa / (self.hyper_prior.kappa + self.prior.kappa)
                                * np.einsum('d,l->dl', self.hyper_prior.gaussian.mu - self.posterior.mu,
                                            self.hyper_prior.gaussian.mu - self.posterior.mu)
                                + xxT - np.einsum('d,l->dl', self.posterior.mu, x) - np.einsum('d,l->dl', x, self.posterior.mu)
                                + n * np.einsum('d,l->dl', self.posterior.mu, self.posterior.mu))
            nu = self.hyper_prior.wishart.nu + n + 1

            self.hyper_posterior.params = rho, kappa, psi, nu

            # vlb.append(self.variational_lowerbound(data))

        _, lmbda = self.hyper_posterior.mean()
        mu = self.posterior.rvs()
        self.likelihood.params = mu, lmbda

        return vlb

    def expected_log_likelihood(self, x):
        raise NotImplementedError

    def variational_lowerbound(self, x):
        # hyp_post_entropy = self.hyper_posterior.wishart.entropy()
        # hyp_post_cross_entropy = self.hyper_posterior.wishart.cross_entropy(self.hyper_prior)
        raise NotImplementedError


class TiedGaussiansWithHierarchicalNormalWisharts:

    def __init__(self, size, dim, hyper_prior, prior):
        self.size = size
        self.dim = dim

        # init hierarchical prior
        taus = np.zeros((self.size, self.dim))
        lmbdas = np.zeros((self.size, self.dim, self.dim))
        for k in range(self.size):
            taus[k], lmbdas[k] = hyper_prior.rvs()

        prior.mus, prior.lmbdas = taus, lmbdas

        self.hyper_prior = hyper_prior
        self.hyper_posterior = copy.deepcopy(self.hyper_prior)

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

        mus = self.prior.rvs(sizes=self.size * [1])
        self.likelihood = TiedGaussiansWithPrecision(size=size, dim=dim,
                                                     mus=mus, lmbdas=lmbdas)

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, data, labels, nb_iter=5):

        lmbdas, mus = None, None
        for _ in range(nb_iter):
            taus = np.zeros((self.size, self.dim))
            lmbdas = np.zeros((self.size, self.dim, self.dim))
            for k in range(self.size):
                taus[k], lmbdas[k] = self.hyper_posterior.rvs()

            self.prior.mus = taus
            self.prior.lmbdas = lmbdas

            # sampling posterior
            xk, nk, xxTk, nk = self.likelihood.weighted_statistics(data, labels)
            self.posterior.nat_param = self.prior.nat_param + Stats([xk, nk])
            self.posterior.lmbdas = lmbdas

            mus = self.posterior.rvs(sizes=self.size * [1])

            # sampling hyper posterior
            rho = np.sum(np.einsum('k,kd->kd', self.prior.kappas, mus)
                         + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu, axis=0) / np.sum(self.prior.kappas + self.hyper_prior.kappa)
            kappa = np.sum(self.prior.kappas + self.hyper_prior.kappa) / self.size
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + np.sum(np.expand_dims(self.hyper_prior.kappa * self.prior.kappas, axis=(1, 2))
                                         * np.einsum('kd,kl->kdl', np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - mus,
                                                     np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - mus) /
                                         (np.expand_dims(self.hyper_prior.kappa + self.prior.kappas, axis=(1, 2))), axis=0) / self.size
                                + np.sum(xxTk, axis=0) / self.size - np.einsum('kd,kl->dl', mus, xk) / self.size
                                - np.einsum('kd,kl->dl', xk, mus) / self.size
                                + np.einsum('k,kd,kl->dl', nk, mus, mus) / self.size)
            nu = np.sum(self.hyper_prior.wishart.nu + nk + 1) / self.size

            self.hyper_posterior.params = rho, kappa, psi, nu

        self.likelihood.mus = mus
        self.likelihood.lmbdas = lmbdas

    # Mean field
    def meanfield_update(self, data, weights, nb_iter=25):

        for _ in range(nb_iter):
            # variational e-step
            xk, nk, xxTk, nk = self.likelihood.weighted_statistics(data, weights)
            self.posterior.kappas = self.prior.kappas + nk
            self.posterior.mus = (np.einsum('k,d->kd', self.prior.kappas, self.hyper_posterior.gaussian.mu) + xk) / (np.expand_dims(self.prior.kappas + nk, axis=1))

            # variational m-step
            rho = np.sum(np.einsum('k,kd->kd', self.prior.kappas, self.posterior.mus)
                         + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu, axis=0) / np.sum(self.prior.kappas + self.hyper_prior.kappa)
            kappa = np.sum(self.prior.kappas + self.hyper_prior.kappa) / self.size
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + np.sum(np.expand_dims(self.hyper_prior.kappa * self.prior.kappas, axis=(1, 2))
                                         * np.einsum('kd,kl->kdl', np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - self.posterior.mus,
                                                     np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - self.posterior.mus) /
                                         (np.expand_dims(self.hyper_prior.kappa + self.prior.kappas, axis=(1, 2))), axis=0) / self.size
                                + np.sum(xxTk, axis=0) / self.size - np.einsum('kd,kl->dl', self.posterior.mus, xk) / self.size
                                - np.einsum('kd,kl->dl', xk, self.posterior.mus) / self.size
                                + np.einsum('k,kd,kl->dl', nk, self.posterior.mus, self.posterior.mus) / self.size)
            nu = np.sum(self.hyper_prior.wishart.nu + nk + 1) / self.size

            self.hyper_posterior.params = rho, kappa, psi, nu

        _, lmbda = self.hyper_posterior.mode()
        mus = self.posterior.mode()
        self.likelihood.mus = mus
        self.likelihood.lmbdas = np.stack(self.size * [lmbda])

    def meanfield_sgdstep(self, data, weights, nb_iter, scale, step_size):

        for _ in range(nb_iter):
            # variational e-step
            tau, lmbda = self.hyper_posterior.mean()

            self.prior.mus = np.stack(self.size * [tau])
            self.prior.lmbdas = np.stack(self.size * [lmbda])

            xk, nk, xxTk, nk = 1. / scale * self.likelihood.weighted_statistics(data, weights)
            self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                       + step_size * (self.prior.nat_param + Stats([xk, nk]))
            self.posterior.lmbdas = np.stack(self.size * [lmbda])

            mus = self.posterior.mean()
            sigmas = np.linalg.inv(self.posterior.omegas)

            # variational m-step
            rho = np.sum(np.einsum('k,kd->kd', self.prior.kappas, mus)
                         + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu, axis=0) / np.sum(self.prior.kappas + self.hyper_prior.kappa)
            kappa = np.sum(self.prior.kappas + self.hyper_prior.kappa) / self.size
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                # + np.sum(np.einsum('k,kdl->kdl', self.prior.kappas, sigmas), axis=0) / self.size
                                # + np.sum(np.einsum('k,kdl->kdl', nk, sigmas), axis=0) / self.size
                                + np.sum(np.expand_dims(self.hyper_prior.kappa * self.prior.kappas, axis=(1, 2))
                                         * np.einsum('kd,kl->kdl', np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - mus,
                                                     np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - mus) /
                                         (np.expand_dims(self.hyper_prior.kappa + self.prior.kappas, axis=(1, 2))), axis=0) / self.size
                                + np.sum(xxTk, axis=0) / self.size - np.einsum('kd,kl->dl', mus, xk) / self.size
                                - np.einsum('kd,kl->dl', xk, mus) / self.size
                                + np.einsum('k,kd,kl->dl', nk, mus, mus) / self.size)
            nu = np.sum(self.hyper_prior.wishart.nu + nk + 1) / self.size

            params = rho, kappa, psi, nu
            self.hyper_posterior.nat_param = (1. - step_size) * self.hyper_posterior.nat_param\
                                             + step_size * (self.hyper_posterior.std_to_nat(params))

        _, lmbda = self.hyper_posterior.mode()
        mus = self.posterior.mode()
        self.likelihood.mus = mus
        self.likelihood.lmbdas = np.stack(self.size * [lmbda])

    def expected_log_likelihood(self, x):
        from scipy.special import digamma

        E_lmbda_mu = np.einsum('dl,kl->kd', self.hyper_posterior.wishart.nu * self.hyper_posterior.wishart.psi, self.posterior.mus)
        E_muT_lmbda_mu = - 0.5 * np.einsum('kd,kd->k', self.posterior.mus, E_lmbda_mu)
        E_lmbda = - 0.5 * (self.hyper_posterior.wishart.nu * self.hyper_posterior.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.hyper_posterior.wishart.nu - np.arange(self.dim)) / 2.))
                                + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.hyper_posterior.wishart.psi_chol))))

        xk, nk, xxTk, nk = self.likelihood.statistics(x, fold=False)
        xxTk += np.expand_dims(np.linalg.inv(self.posterior.omegas), axis=1)

        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kd,knd->kn', E_lmbda_mu, xk)\
               + np.einsum('k,kn->kn', E_muT_lmbda_mu, nk)\
               + np.einsum('dl,kndl->kn', E_lmbda, xxTk)\
               + E_logdet_lmbda * nk

    def variational_lowerbound(self):
        from scipy.special import digamma

        vlb = 0.
        for k in range(self.size):
            # entropy of hyper posterior
            vlb += self.hyper_posterior.entropy()

            # cross entropy of hyper posterior
            vlb += - self.hyper_posterior.cross_entropy(self.hyper_prior)

            # entropy of posterior
            vlb += self.posterior.dists[k].entropy()

            # expected cross entropy of posterior
            vlb += - 0.5 * self.dim * np.log(2. * np.pi)

            E_logdet_lmbda = np.sum(digamma((self.hyper_posterior.wishart.nu - np.arange(self.dim)) / 2.))\
                             + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.hyper_posterior.wishart.psi_chol)))

            vlb += 0.5 * self.dim * np.log(self.prior.kappas[k])
            vlb += 0.5 * E_logdet_lmbda
            vlb += - 0.5 * self.prior.kappas[k] * self.dim / self.hyper_posterior.kappa
            vlb += - 0.5 * np.einsum('d,dl,l->', self.posterior.mus[k] - self.hyper_posterior.gaussian.mu,
                                     self.prior.kappas[k] * self.hyper_posterior.wishart.nu
                                     * self.hyper_posterior.wishart.psi,
                                     self.posterior.mus[k] - self.hyper_posterior.gaussian.mu)
            vlb += - 0.5 * np.trace(self.prior.kappas[k] * self.hyper_posterior.wishart.nu
                                    * self.hyper_posterior.wishart.psi @ np.linalg.inv(self.posterior.omegas[k]))

        return vlb

    def posterior_predictive_gaussian(self):
        mus, omegas = self.posterior.mus, self.posterior.omegas
        _, _, psis, nus = list([np.stack(self.size * [p])
                                for p in self.hyper_posterior.params])
        dfs = nus - self.dim + 1
        lmbdas = np.einsum('k,kdl->kdl', dfs, psis)
        return mus, lmbdas

    def log_posterior_predictive_gaussian(self, x):
        mus, lmbdas = self.posterior_predictive_gaussian()
        return stacked_mvn_logpdf(x, mus, lmbdas)


class LinearGaussianWithMatrixNormalWishart:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal Wishart prior.
    """

    def __init__(self, column_dim, row_dim, prior,
                 likelihood=None, affine=True):
        # Matrix-Normal-Wishart prior
        self.prior = prior

        # Matrix-Normal-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        # Linear Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda = self.prior.rvs()
            self.likelihood = LinearGaussianWithPrecision(column_dim, row_dim,
                                                          A=A, lmbda=lmbda,
                                                          affine=affine)

    def empirical_bayes(self, x, y):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, x, y, weights=None):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, x, y, z=None):
        stats = self.likelihood.statistics(x, y) if z is None\
            else self.likelihood.weighted_statistics(x, y, z)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, x, y, weights=None):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, x, y, weights, scale, step_size):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                   + step_size * (self.prior.nat_param + 1. / scale * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x, y):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a linear Gaussian likelihood
        # log-parition is subsumed into nat*stats
        stats = self.likelihood.statistics(x, y, fold=False)
        log_base = self.likelihood.log_base()

        return log_base\
               + np.einsum('dl,ndl->n', nat_param[0], stats[0])\
               + np.einsum('dl,ndl->n', nat_param[1], stats[1])\
               + np.einsum('dl,ndl->n', nat_param[2], stats[2])\
               + nat_param[3] * stats[3]

    def posterior_predictive_gaussian(self, x):
        x = np.reshape(x, (-1, self.likelihood.input_dim))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.row_dim + 1
        mu = np.einsum('dl,nl->nd', M, x)

        c = 1. + np.einsum('nd,dl,nl->n', x, np.linalg.inv(K), x)
        lmbda = np.einsum('dl,n->ndl', psi, df / c)
        return mu, lmbda

    def log_posterior_predictive_gaussian(self, x, y):
        mu, lmbda = self.posterior_predictive_gaussian(x)
        return mvn_logpdf(y, mu, lmbda)

    def posterior_predictive_studentt(self, x):
        x = np.reshape(x, (-1, self.likelihood.input_dim))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.row_dim + 1
        mu = np.einsum('dl,nl->nd', M, x)

        c = 1. + np.einsum('nd,dl,nl->n', x, np.linalg.inv(K), x)
        lmbda = np.einsum('dl,n->ndl', psi, df / c)
        return mu, lmbda, df

    def log_posterior_predictive_studentt(self, x, y):
        mu, lmbda, df = self.posterior_predictive_studentt(x)
        return mvt_logpdf(y, mu, lmbda, df)


class StackedLinearGaussiansWithMatrixNormalWisharts(LinearGaussianWithMatrixNormalWishart, ABC):

    def __init__(self, size, column_dim, row_dim,
                 prior, likelihood=None, affine=True):

        self.size = size

        # Linear Gaussian likelihood
        if likelihood is None:
            As, lmbdas = prior.rvs()
            likelihood = StackedLinearGaussiansWithPrecision(size, column_dim, row_dim,
                                                             As=As, lmbdas=lmbdas,
                                                             affine=affine)

        super(StackedLinearGaussiansWithMatrixNormalWisharts, self).__init__(column_dim, row_dim,
                                                                             prior, likelihood)

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x, y):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a linear Gaussian likelihood
        # log-parition is subsumed into nat*stats
        stats = self.likelihood.statistics(x, y, fold=False)
        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kdl,kndl->kn', nat_param[0], stats[0])\
               + np.einsum('kdl,kndl->kn', nat_param[1], stats[1])\
               + np.einsum('kdl,kndl->kn', nat_param[2], stats[2])\
               + np.einsum('k,kn->kn', nat_param[3], stats[3])

    def posterior_predictive_gaussian(self, x):
        x = np.reshape(x, (-1, self.likelihood.input_dim))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        Ms, Ks, psis, nus = self.posterior.params

        dfs = nus - self.likelihood.row_dim + 1
        mus = np.einsum('kdl,nl->knd', Ms, x)

        cs = 1. + np.einsum('nd,kdl,nl->kn', x, np.linalg.inv(Ks), x)
        lmbdas = np.einsum('kdl,k,kn->kndl', psis, dfs, 1. / cs)
        return mus, lmbdas

    def log_posterior_predictive_gaussian(self, x, y):
        mus, lmbdas = self.posterior_predictive_gaussian(x)
        return stacked_mvn_logpdf(y, mus, lmbdas)

    def posterior_predictive_studentt(self, x):
        x = np.reshape(x, (-1, self.likelihood.input_dim))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        Ms, Ks, psis, nus = self.posterior.params

        dfs = nus - self.likelihood.row_dim + 1
        mus = np.einsum('kdl,nl->knd', Ms, x)

        cs = 1. + np.einsum('nd,kdl,nl->kn', x, np.linalg.inv(Ks), x)
        lmbdas = np.einsum('kdl,k,kn->kndl', psis, dfs, 1. / cs)
        return mus, lmbdas, dfs

    def log_posterior_predictive_studentt(self, x, y):
        mus, lmbdas, dfs = self.posterior_predictive_studentt(x)
        return stacked_mvt_logpdf(y, mus, lmbdas, dfs)


class TiedLinearGaussiansWithMatrixNormalWisharts(StackedLinearGaussiansWithMatrixNormalWisharts, ABC):

    def __init__(self, size, column_dim, row_dim,
                 prior, likelihood=None, affine=True):

        self.size = size

        # tied linear Gaussian likelihood
        if likelihood is None:
            As, lmbdas = prior.rvs()
            likelihood = TiedLinearGaussiansWithPrecision(size, column_dim, row_dim,
                                                          As=As, lmbdas=lmbdas,
                                                          affine=affine)

        super(TiedLinearGaussiansWithMatrixNormalWisharts, self).__init__(size, column_dim, row_dim,
                                                                          prior, likelihood)


class LinearGaussianWithMatrixNormalGamma:

    def __init__(self, column_dim, row_dim, prior,
                 likelihood=None, affine=True):

        # Matrix-Normal-Gamma prior
        self.prior = prior

        # Matrix-Normal-Gamma posterior
        self.posterior = copy.deepcopy(prior)

        # Diagonal Linear Gaussian likelihood
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            A, lmbda_diag = self.prior.rvs()
            self.likelihood = LinearGaussianWithDiagonalPrecision(column_dim, row_dim,
                                                                  A=A, lmbda_diag=lmbda_diag,
                                                                  affine=affine)

    def empirical_bayes(self, x, y):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, x, y, weights=None):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, x, y, z=None):
        stats = self.likelihood.statistics(x, y) if z is None\
            else self.likelihood.weighted_statistics(x, y, z)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, x, y, weights=None):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, x, y, weights, scale, step_size):
        stats = self.likelihood.statistics(x, y) if weights is None\
            else self.likelihood.weighted_statistics(x, y, weights)
        self.posterior.nat_param = (1. - step_size) * self.posterior.nat_param\
                                   + step_size * (self.prior.nat_param + 1. / scale * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x, y):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a diagonal linear Gaussian likelihood
        # log-parition is subsumed into nat*stats
        stats = self.likelihood.statistics(x, y, fold=False)
        log_base = self.likelihood.log_base()

        return log_base\
               + np.einsum('dl,ndl->n', nat_param[0], stats[0])\
               + np.einsum('dl,ndl->n', nat_param[1], stats[1])\
               + np.einsum('d,nd->n', nat_param[2], stats[2])\
               + np.einsum('d,nd->n', nat_param[3], stats[3])


class StackedLinearGaussiansWithMatrixNormalGammas(LinearGaussianWithMatrixNormalGamma, ABC):

    def __init__(self, size, column_dim, row_dim,
                 prior, likelihood=None, affine=True):

        self.size = size

        # Diagonal Linear Gaussian likelihood
        if likelihood is None:
            As, lmbdas_diags = prior.rvs()
            likelihood = StackedLinearGaussiansWithDiagonalPrecision(size, column_dim, row_dim,
                                                                     As=As, lmbdas_diags=lmbdas_diags,
                                                                     affine=affine)

        super(StackedLinearGaussiansWithMatrixNormalGammas, self).__init__(column_dim, row_dim,
                                                                           prior, likelihood)

    # expected log_likelihood under posterior
    def expected_log_likelihood(self, x, y):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.posterior.expected_statistics()

        # Data statistics under a diagonal linear Gaussian likelihood
        # log-parition is subsumed into nat*stats
        stats = self.likelihood.statistics(x, y, fold=False)
        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kdl,kndl->kn', nat_param[0], stats[0])\
               + np.einsum('kdl,kndl->kn', nat_param[1], stats[1])\
               + np.einsum('kd,knd->kn', nat_param[2], stats[2])\
               + np.einsum('kd,knd->kn', nat_param[3], stats[3])


class TiedLinearGaussiansWithMatrixNormalGammas(StackedLinearGaussiansWithMatrixNormalGammas, ABC):

    def __init__(self, size, column_dim, row_dim,
                 prior, likelihood=None, affine=True):

        self.size = size

        # tied linear Gaussian likelihood
        if likelihood is None:
            As, lmbdas_diags = prior.rvs()
            likelihood = TiedLinearGaussiansWithDiagonalPrecision(size, column_dim, row_dim,
                                                                  As=As, lmbdas_diags=lmbdas_diags,
                                                                  affine=affine)

        super(TiedLinearGaussiansWithMatrixNormalGammas, self).__init__(size, column_dim, row_dim,
                                                                        prior, likelihood)


class AffineLinearGaussianWithMatrixNormalWishart:

    def __init__(self, column_dim, row_dim,
                 slope_prior, offset_prior, precision_prior,
                 likelihood=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        lmbda = precision_prior.rvs()
        slope_prior.V = lmbda
        A = slope_prior.rvs()

        offset_prior.lmbda = lmbda
        c = offset_prior.rvs()

        self.slope_prior = slope_prior
        self.offset_prior = offset_prior
        self.precision_prior = precision_prior

        self.likelihood = AffineLinearGaussianWithPrecision(column_dim, row_dim,
                                                            A, c, lmbda)

        self.slope_posterior = copy.deepcopy(self.slope_prior)
        self.offset_posterior = copy.deepcopy(self.offset_prior)
        self.precision_posterior = copy.deepcopy(self.precision_prior)

    def empirical_bayes(self, data):
        raise NotImplementedError

    def resample(self, x, y, nb_iter=25):

        lmbda, A, c = None, None, None
        for _ in range(nb_iter):
            c = self.offset_posterior.rvs()

            xm = np.einsum('nd->d', x, optimize=True)
            ym = np.einsum('nd->d', y, optimize=True)
            n = y.shape[0]

            yxT = np.einsum('nd,nl->dl', y, x, optimize=True)
            xxT = np.einsum('nd,nl->dl', x, x, optimize=True)
            yyT = np.einsum('nd,nl->dl', y, y, optimize=True)

            ccT = np.einsum('d,l->dl', c, c, optimize=True)
            ycT = np.einsum('nd,l->dl', y, c, optimize=True)
            cxT = np.einsum('d,nl->dl', c, x, optimize=True)

            M0, K0 = self.slope_prior.M, self.slope_prior.K
            psi0, nu0 = self.precision_prior.psi, self.precision_prior.nu

            M = (M0 @ K0 + yxT - cxT) @ np.linalg.inv(K0 + xxT)
            K = K0 + xxT

            self.slope_posterior.M = M
            self.slope_posterior.K = K

            psi = np.linalg.inv(np.linalg.inv(psi0)
                                + M0 @ K0 @ M0.T
                                + np.sum(np.einsum('nd,nl->ndl', y - c, y - c), axis=0)
                                + self.offset_prior.kappa * np.einsum('d,l->dl', c - self.offset_prior.mu,
                                                                                 c - self.offset_prior.mu)
                                - M @ K @ M.T)

            nu = nu0 + n + 1

            self.precision_posterior.psi = psi
            self.precision_posterior.nu = nu

            lmbda = self.precision_posterior.rvs()
            self.slope_posterior.V = lmbda
            A = self.slope_posterior.rvs()

            rho = (self.offset_prior.kappa * self.offset_prior.mu + (ym - A @ xm)) / (self.offset_prior.kappa + n)
            kappa = self.offset_prior.kappa + n

            self.offset_posterior.mu = rho
            self.offset_posterior.kappa = kappa
            self.offset_posterior.lmbda = lmbda

        self.likelihood.A = A
        self.likelihood.c = c
        self.likelihood.lmbda = lmbda


class TiedAffineLinearGaussiansWithMatrixNormalWisharts:
    # Linear Gaussians sharing the same slope

    def __init__(self, size, column_dim, row_dim,
                 slope_prior, offset_prior, precision_prior,
                 likelihood=None):

        self.size = size
        self.column_dim = column_dim
        self.row_dim = row_dim

        As = np.zeros((self.size, self.row_dim, self.column_dim))
        lmbdas = np.zeros((self.size, self.row_dim, self.row_dim))
        for k in range(self.size):
            lmbdas[k] = precision_prior.rvs()
            slope_prior.V = lmbdas[k]
            As[k] = slope_prior.rvs()

        offset_prior.lmbdas = lmbdas
        cs = offset_prior.rvs(sizes=self.size * [1])

        self.slope_prior = slope_prior
        self.offset_prior = offset_prior
        self.precision_prior = precision_prior

        self.likelihood = StackedAffineLinearGaussiansWithPrecision(size,
                                                                    column_dim, row_dim,
                                                                    As, cs, lmbdas)

        self.slope_posterior = copy.deepcopy(self.slope_prior)
        self.offset_posterior = copy.deepcopy(self.offset_prior)
        self.precision_posterior = copy.deepcopy(self.precision_prior)

    def empirical_bayes(self, x, y):
        raise NotImplementedError

    def resample(self, x, y, z, nb_iter=25):

        lmbdas, As, cs = None, None, None
        for _ in range(nb_iter):
            cs = self.offset_posterior.rvs(sizes=self.size * [1])

            xmk = np.einsum('kn,nd->kd', z, x, optimize=True)
            ymk = np.einsum('kn,nd->kd', z, y, optimize=True)
            nk = np.sum(z, axis=1)

            yxTk = np.einsum('nd,kn,nl->kdl', y, z, x, optimize=True)
            xxTk = np.einsum('nd,kn,nl->kdl', x, z, x, optimize=True)
            yyTk = np.einsum('nd,kn,nl->kdl', y, z, y, optimize=True)

            ccTk = np.einsum('kd,kn,kl->kdl', cs, z, cs, optimize=True)
            ycTk = np.einsum('nd,kn,kl->kdl', y, z, cs, optimize=True)
            cxTk = np.einsum('kd,kn,nl->kdl', cs, z, x, optimize=True)

            M0, K0 = self.slope_prior.M, self.slope_prior.K
            psi0, nu0 = self.precision_prior.psi, self.precision_prior.nu

            M = np.sum(np.einsum('kdl,klh->kdh', np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk,
                                                 np.linalg.inv(np.expand_dims(K0, axis=0) + xxTk)), axis=0) / self.size
            K = np.sum(np.expand_dims(K0, axis=0) + xxTk, axis=0) / self.size

            self.slope_posterior.M = M
            self.slope_posterior.K = K

            psi = np.linalg.inv(np.linalg.inv(psi0)
                                + M0 @ K @ M0.T
                                + np.sum(np.einsum('kn,knd,knl->kdl', z, np.expand_dims(y, axis=0) - np.expand_dims(cs, axis=1),
                                                                         np.expand_dims(y, axis=0) - np.expand_dims(cs, axis=1)), axis=0) / self.size
                                + np.sum(np.einsum('k,kd,kl->kdl', self.offset_prior.kappas,
                                                                   cs - self.offset_prior.mus,
                                                                   cs - self.offset_prior.mus), axis=0) / self.size
                                - np.sum(np.einsum('kdl,klm,khm->kdh', np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk,
                                                                       np.linalg.inv(np.expand_dims(K0, axis=0) + xxTk),
                                                                       np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk), axis=0) / self.size)
            nu = np.sum(nu0 + nk + 1) / self.size

            self.precision_posterior.psi = psi
            self.precision_posterior.nu = nu

            As = np.zeros((self.size, self.row_dim, self.column_dim))
            lmbdas = np.zeros((self.size, self.row_dim, self.row_dim))
            for k in range(self.size):
                lmbdas[k] = self.precision_posterior.rvs()
                self.slope_posterior.V = lmbdas[k]
                As[k] = self.slope_posterior.rvs()

            rhos = np.einsum('k,kd->kd', 1. / (self.offset_prior.kappas + nk),
                                        np.einsum('k,kd->kd', self.offset_prior.kappas, self.offset_prior.mus)
                                        + (ymk - np.einsum('kdl,kl->kd', As, xmk)))
            kappas = self.offset_prior.kappas + nk

            self.offset_posterior.mus = rhos
            self.offset_posterior.kappas = kappas
            self.offset_posterior.lmbdas = lmbdas

        self.likelihood.As = As
        self.likelihood.cs = cs
        self.likelihood.lmbdas = lmbdas

    def meanfield_update(self, x, y, weights, nb_iter=25):

        for _ in range(nb_iter):
            cs = self.offset_posterior.mean()
            sigmas = np.linalg.inv(self.offset_posterior.omegas)

            xmk = np.einsum('kn,nd->kd', weights, x, optimize=True)
            ymk = np.einsum('kn,nd->kd', weights, y, optimize=True)
            nk = np.sum(weights, axis=1)

            yxTk = np.einsum('nd,kn,nl->kdl', y, weights, x, optimize=True)
            xxTk = np.einsum('nd,kn,nl->kdl', x, weights, x, optimize=True)
            yyTk = np.einsum('nd,kn,nl->kdl', y, weights, y, optimize=True)

            ccTk = np.einsum('kd,kn,kl->kdl', cs, weights, cs, optimize=True)
            ycTk = np.einsum('nd,kn,kl->kdl', y, weights, cs, optimize=True)
            cxTk = np.einsum('kd,kn,nl->kdl', cs, weights, x, optimize=True)

            M0, K0 = self.slope_prior.M, self.slope_prior.K
            psi0, nu0 = self.precision_prior.psi, self.precision_prior.nu

            M = np.sum(np.einsum('kdl,klh->kdh', np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk,
                                 np.linalg.inv(np.expand_dims(K0, axis=0) + xxTk)), axis=0) / self.size
            K = np.sum(np.expand_dims(K0, axis=0) + xxTk, axis=0) / self.size

            self.slope_posterior.M = M
            self.slope_posterior.K = K

            psi = np.linalg.inv(np.linalg.inv(psi0)
                                + M0 @ K @ M0.T
                                # + np.sum(np.einsum('k,kdl->kdl', self.offset_prior.kappas, sigmas), axis=0) / self.size
                                # + np.sum(np.einsum('k,kdl->kdl', nk, sigmas), axis=0) / self.size
                                + np.sum(np.einsum('kn,knd,knl->kdl', weights, np.expand_dims(y, axis=0) - np.expand_dims(cs, axis=1),
                                                   np.expand_dims(y, axis=0) - np.expand_dims(cs, axis=1)), axis=0) / self.size
                                + np.sum(np.einsum('k,kd,kl->kdl', self.offset_prior.kappas,
                                                   cs - self.offset_prior.mus,
                                                   cs - self.offset_prior.mus), axis=0) / self.size
                                - np.sum(np.einsum('kdl,klm,khm->kdh', np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk,
                                                   np.linalg.inv(np.expand_dims(K0, axis=0) + xxTk),
                                                   np.expand_dims(M0 @ K0, axis=0) + yxTk - cxTk), axis=0) / self.size)
            nu = np.sum(nu0 + nk + 1) / self.size

            self.precision_posterior.psi = psi
            self.precision_posterior.nu = nu

            lmbda = self.precision_posterior.mean()
            self.slope_posterior.V = lmbda
            A = self.slope_posterior.mean()

            rhos = np.einsum('k,kd->kd', 1. / (self.offset_prior.kappas + nk),
                                         np.einsum('k,kd->kd', self.offset_prior.kappas, self.offset_prior.mus)
                                         + (ymk - np.einsum('dl,kl->kd', A, xmk)))
            kappas = self.offset_prior.kappas + nk

            self.offset_posterior.mus = rhos
            self.offset_posterior.kappas = kappas
            self.offset_posterior.lmbdas = np.stack(self.size * [lmbda])

        A = self.slope_posterior.mode()
        lmbda = self.precision_posterior.mode()
        cs = self.offset_posterior.mode()

        self.likelihood.As = np.stack(self.size * [A])
        self.likelihood.lmbdas = np.stack(self.size * [lmbda])
        self.likelihood.cs = cs

    def expected_log_likelihood(self, x, y):
        import scipy as sc
        from scipy import linalg

        from mimo.distributions import StackedMatrixNormalWisharts

        M0 = np.zeros((self.size, self.row_dim, self.column_dim + 1))
        K0 = np.zeros((self.size, self.column_dim + 1, self.column_dim + 1))
        for k in range(self.size):
            M0[k] = np.hstack((self.slope_prior.M, self.offset_prior.mus[k][:, None]))
            K0[k] = sc.linalg.block_diag(self.slope_prior.K,
                                         np.diag(np.array([self.offset_prior.kappas[k]])))

        psi0 = np.stack(self.size * [self.precision_prior.psi])
        nu0 = np.stack(self.size * [self.precision_prior.nu])

        prior = StackedMatrixNormalWisharts(size=self.size,
                                            column_dim=self.column_dim + 1,
                                            row_dim=self.row_dim,
                                            Ms=M0, Ks=K0, psis=psi0, nus=nu0)

        dist = StackedLinearGaussiansWithMatrixNormalWisharts(size=self.size,
                                                              column_dim=self.column_dim + 1,
                                                              row_dim=self.row_dim,
                                                              prior=prior, affine=True)

        for k in range(self.size):
            dist.posterior.dists[k].matnorm.M = np.hstack((self.slope_posterior.M, self.offset_posterior.mus[k][:, None]))
            dist.posterior.dists[k].matnorm.K = sc.linalg.block_diag(self.slope_posterior.K,
                                                                     np.diag(np.array([self.offset_posterior.kappas[k]])))
            dist.posterior.dists[k].wishart.psi = self.precision_posterior.psi
            dist.posterior.dists[k].wishart.nu = self.precision_posterior.nu

        return dist.expected_log_likelihood(x, y)

        # another valid implementation

        # from scipy.special import digamma
        #
        # E_Lmbda_c = np.einsum('dl,kl->kd', self.precision_posterior.nu * self.precision_posterior.psi, self.offset_posterior.mus)
        # E_cT_Lmbda_c = - 0.5 * (self.row_dim / self.offset_posterior.kappas + np.einsum('kd,kd->k', self.offset_posterior.mus, E_Lmbda_c))
        #
        # E_Lmbda_A = self.precision_posterior.nu * self.precision_posterior.psi @ self.slope_posterior.M
        # E_AT_Lmbda_A = - 0.5 * (self.row_dim * np.linalg.inv(self.slope_posterior.K) + self.slope_posterior.M.T.dot(E_Lmbda_A))
        #
        # E_cT_Lmbda_A = - np.einsum('kd,dl->kl', self.offset_posterior.mus, E_Lmbda_A)
        #
        # E_lmbda = - 0.5 * (self.precision_posterior.nu * self.precision_posterior.psi)
        # E_logdet_lmbda = 0.5 * (np.sum(digamma((self.precision_posterior.nu - np.arange(self.row_dim)) / 2.))
        #                         + self.row_dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.precision_posterior.psi_chol))))
        #
        # ymk, xmk, yxTk, xxTk, yyTk, nk = self.likelihood.statistics(x, y, fold=False)
        #
        # log_base = self.likelihood.log_base()
        # return np.expand_dims(log_base, axis=1)\
        #         + np.einsum('dl,kndl->kn', E_AT_Lmbda_A, xxTk)\
        #         + np.einsum('dl,kndl->kn', E_Lmbda_A, yxTk)\
        #         + np.einsum('dl,kndl->kn', E_lmbda, yyTk)\
        #         + E_logdet_lmbda * nk \
        #         + np.einsum('kd,knd->kn', E_Lmbda_c, ymk)\
        #         + np.einsum('kd,knd->kn', E_cT_Lmbda_A, xmk)\
        #         + np.einsum('k,kn->kn', E_cT_Lmbda_c, nk)

    def variational_lowerbound(self):
        import scipy as sc
        from scipy import linalg

        from mimo.distributions import StackedMatrixNormalWisharts

        M0 = np.zeros((self.size, self.row_dim, self.column_dim + 1))
        K0 = np.zeros((self.size, self.column_dim + 1, self.column_dim + 1))
        for k in range(self.size):
            M0[k] = np.hstack((self.slope_prior.M, self.offset_prior.mus[k][:, None]))
            K0[k] = sc.linalg.block_diag(self.slope_prior.K,
                                         np.diag(np.array([self.offset_prior.kappas[k]])))

        psi0 = np.stack(self.size * [self.precision_prior.psi])
        nu0 = np.stack(self.size * [self.precision_prior.nu])

        prior = StackedMatrixNormalWisharts(size=self.size,
                                            column_dim=self.column_dim + 1,
                                            row_dim=self.row_dim,
                                            Ms=M0, Ks=K0, psis=psi0, nus=nu0)

        posterior = copy.deepcopy(prior)

        for k in range(self.size):
            posterior.dists[k].matnorm.M = np.hstack((self.slope_posterior.M, self.offset_posterior.mus[k][:, None]))
            posterior.dists[k].matnorm.K = sc.linalg.block_diag(self.slope_posterior.K,
                                                                np.diag(np.array([self.offset_posterior.kappas[k]])))
            posterior.dists[k].wishart.psi = self.precision_posterior.psi
            posterior.dists[k].wishart.nu = self.precision_posterior.nu

        q_entropy = posterior.entropy()
        qp_cross_entropy = posterior.cross_entropy(prior)
        return q_entropy - qp_cross_entropy

    def posterior_predictive_gaussian(self, x):
        import scipy as sc
        from scipy import linalg

        from mimo.distributions import StackedMatrixNormalWisharts

        M0 = np.zeros((self.size, self.row_dim, self.column_dim + 1))
        K0 = np.zeros((self.size, self.column_dim + 1, self.column_dim + 1))
        for k in range(self.size):
            M0[k] = np.hstack((self.slope_prior.M, self.offset_prior.mus[k][:, None]))
            K0[k] = sc.linalg.block_diag(self.slope_prior.K,
                                         np.diag(np.array([self.offset_prior.kappas[k]])))

        psi0 = np.stack(self.size * [self.precision_prior.psi])
        nu0 = np.stack(self.size * [self.precision_prior.nu])

        prior = StackedMatrixNormalWisharts(size=self.size,
                                            column_dim=self.column_dim + 1,
                                            row_dim=self.row_dim,
                                            Ms=M0, Ks=K0, psis=psi0, nus=nu0)

        dist = StackedLinearGaussiansWithMatrixNormalWisharts(size=self.size,
                                                              column_dim=self.column_dim + 1,
                                                              row_dim=self.row_dim,
                                                              prior=prior, affine=True)

        for k in range(self.size):
            dist.posterior.dists[k].matnorm.M = np.hstack((self.slope_posterior.M, self.offset_posterior.mus[k][:, None]))
            dist.posterior.dists[k].matnorm.K = sc.linalg.block_diag(self.slope_posterior.K,
                                                                     np.diag(np.array([self.offset_posterior.kappas[k]])))
            dist.posterior.dists[k].wishart.psi = self.precision_posterior.psi
            dist.posterior.dists[k].wishart.nu = self.precision_posterior.nu

        return dist.posterior_predictive_gaussian(x)

    def log_posterior_predictive_gaussian(self, x, y):
        mus, lmbdas = self.posterior_predictive_gaussian(x)
        return stacked_mvn_logpdf(y, mus, lmbdas)
