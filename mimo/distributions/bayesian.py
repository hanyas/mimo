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

from mimo.utils.stats import multivariate_gaussian_loglik as mvn_logpdf
from mimo.utils.stats import multivariate_studentt_loglik as mvt_logpdf

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

    def meanfield_sgdstep(self, data, weights, scale, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / scale * stats)

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

    def meanfield_sgdstep(self, data, weights, scale, stepsize):
        counts = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        acc_counts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = (1. - stepsize) * self.posterior.gammas\
                                + stepsize * (self.prior.gammas + 1. / scale * counts)
        self.posterior.deltas = (1. - stepsize) * self.posterior.deltas\
                                + stepsize * (self.prior.deltas + 1. / scale * acc_counts)

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

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

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

    def posterior_predictive_gaussian(self):
        mu, kappa, psi, nu = self.posterior.params
        df = nu - self.dim + 1
        c = 1. + 1. / kappa
        lmbda = df * psi / c
        return mu, lmbda

    def log_posterior_predictive_gaussian(self, x):
        mu, lmbda = self.posterior_predictive_gaussian()
        return mvn_logpdf(x, mu, lmbda)

    def posterior_predictive_studentt(self):
        mu, kappa, psi, nu = self.posterior.params
        df = nu - self.dim + 1
        c = 1. + 1. / kappa
        lmbda = df * psi / c
        return mu, lmbda, df

    def log_posterior_predictive_studentt(self, x):
        mu, lmbda, df = self.posterior_predictive_studentt()
        return mvt_logpdf(x, mu, lmbda, df)


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

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.likelihood.statistics(data) if weights is None\
            else self.likelihood.weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

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

    def posterior_predictive_gaussian(self):
        mu, kappas, alphas, betas = self.posterior.params
        c = 1. + 1. / kappas
        lmbda_diag = (alphas / betas) * 1. / c
        return mu, lmbda_diag

    def log_posterior_predictive_gaussian(self, x):
        mu, lmbda_diag = self.posterior_predictive_gaussian()
        log_posterior = 0.
        for _x, _mu, _lmbda in zip(x, mu, lmbda_diag):
            log_posterior += mvn_logpdf(_x.reshape(-1, 1),
                                        _mu.reshape(-1, 1),
                                        _lmbda.reshape(-1, 1, 1))
        return log_posterior

    def posterior_predictive_studentt(self):
        mu, kappas, alphas, betas = self.posterior.params
        dfs = 2. * alphas
        c = 1. + 1. / kappas
        lmbda_diag = (alphas / betas) * 1. / c
        return mu, lmbda_diag, dfs

    def log_posterior_predictive_studentt(self, x):
        mu, lmbda_diag, dfs = self.posterior_predictive_studentt()
        log_posterior = 0.
        for _x, _mu, _lmbda, _df in zip(x, mu, lmbda_diag, dfs):
            log_posterior += mvt_logpdf(_x.reshape(-1, 1),
                                        _mu.reshape(-1, 1),
                                        _lmbda.reshape(-1, 1, 1),
                                        _df)
        return log_posterior


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
            # sampling posterior
            x, n, xxT, n = self.likelihood.statistics(data)
            self.posterior.nat_param = self.prior.nat_param + Stats([x, n])

            mu = self.posterior.rvs()

            # sampling hyper posterior
            # stats = Stats([self.prior.kappa * mu,
            #                self.prior.kappa,
            #                xxT - np.outer(mu, x) - np.outer(x, mu)
            #                + (n + self.prior.kappa) * np.outer(mu, mu),
            #                n + 1])
            #
            # self.hyper_posterior.nat_param = self.hyper_prior + stats

            rho = (self.prior.kappa * mu + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu)\
                  / (self.prior.kappa + self.hyper_prior.kappa)
            kappa = self.prior.kappa + self.hyper_prior.kappa
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + self.hyper_prior.kappa * self.prior.kappa / (self.hyper_prior.kappa + self.prior.kappa)
                                * np.einsum('d,l->dl', self.hyper_prior.gaussian.mu - mu, self.hyper_prior.gaussian.mu - mu)
                                + xxT - np.einsum('d,l->dl', mu, x) - np.einsum('d,l->dl', x, mu) + n * np.einsum('d,l->dl', mu, mu))
            nu = self.hyper_prior.wishart.nu + n + 1

            self.hyper_posterior.params = rho, kappa, psi, nu

            tau, lmbda = self.hyper_posterior.rvs()

            self.prior.mu, self.prior.lmbda = tau, lmbda
            self.posterior.lmbda = lmbda

        self.likelihood.params = mu, lmbda

    # Mean field
    def meanfield_update(self, data, nb_iter=25):

        vlb = []
        for i in range(nb_iter):
            # variational e-step
            tau, lmbda = self.hyper_posterior.mean()

            self.prior.mu = tau
            self.posterior.lmbda = lmbda

            x, n, xxT, n = self.likelihood.statistics(data)
            self.posterior.nat_param = self.prior.nat_param + Stats([x, n])

            # variational m-step
            mu = (self.prior.kappa * self.posterior.mu + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu)\
                 / (self.prior.kappa + self.hyper_prior.kappa)
            kappa = self.prior.kappa + self.hyper_prior.kappa
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + (self.prior.kappa + n) * np.linalg.inv(self.posterior.omega)
                                + self.hyper_prior.kappa * self.prior.kappa / (self.hyper_prior.kappa + self.prior.kappa)
                                * np.einsum('d,l->dl', self.hyper_prior.gaussian.mu - self.posterior.mu,
                                            self.hyper_prior.gaussian.mu - self.posterior.mu)
                                + xxT - np.einsum('d,l->dl', self.posterior.mu, x) - np.einsum('d,l->dl', x, self.posterior.mu)
                                + n * np.einsum('d,l->dl', self.posterior.mu, self.posterior.mu))
            nu = self.hyper_prior.wishart.nu + n + 1

            self.hyper_posterior.params = mu, kappa, psi, nu

            # vlb.append(self.variational_lowerbound(data))

        _, lmbda = self.hyper_posterior.mode()
        mu = self.posterior.mode()
        self.likelihood.params = mu, lmbda

        return vlb

    def expected_log_likelihood(self, x):
        raise NotImplementedError

    def variational_lowerbound(self, x):
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
            # sampling posterior
            xk, nk, xxTk, nk = self.likelihood.weighted_statistics(data, labels)
            self.posterior.nat_param = self.prior.nat_param + Stats([xk, nk])

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

            taus = np.zeros((self.size, self.dim))
            lmbdas = np.zeros((self.size, self.dim, self.dim))
            for k in range(self.size):
                taus[k], lmbdas[k] = self.hyper_posterior.rvs()

            self.prior.mus = taus
            self.prior.lmbdas = lmbdas
            self.posterior.lmbdas = lmbdas

        self.likelihood.mus = mus
        self.likelihood.lmbdas = lmbdas

    # Mean field
    def meanfield_update(self, data, weights, nb_iter=25):

        for i in range(nb_iter):
            # variational e-step
            tau, lmbda = self.hyper_posterior.mean()

            self.prior.mus = np.stack(self.size * [tau])
            self.prior.lmbdas = np.stack(self.size * [lmbda])
            self.posterior.lmbdas = np.stack(self.size * [lmbda])

            xk, nk, xxTk, nk = self.likelihood.weighted_statistics(data, weights)
            self.posterior.nat_param = self.prior.nat_param + Stats([xk, nk])

            # variational m-step
            mu = np.sum(np.einsum('k,kd->kd', self.prior.kappas, self.posterior.mus)
                        + self.hyper_prior.kappa * self.hyper_prior.gaussian.mu, axis=0) / np.sum(self.prior.kappas + self.hyper_prior.kappa)
            kappa = np.sum(self.prior.kappas + self.hyper_prior.kappa) / self.size
            psi = np.linalg.inv(np.linalg.inv(self.hyper_prior.wishart.psi)
                                + np.sum(np.einsum('k,kdl->kdl', self.prior.kappas, np.linalg.inv(self.posterior.omegas)), axis=0) / self.size
                                + np.sum(np.einsum('k,kdl->kdl', nk, np.linalg.inv(self.posterior.omegas)), axis=0) / self.size
                                + np.sum(np.expand_dims(self.hyper_prior.kappa * self.prior.kappas, axis=(1, 2))
                                         * np.einsum('kd,kl->kdl', np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - self.posterior.mus,
                                                     np.expand_dims(self.hyper_prior.gaussian.mu, axis=0) - self.posterior.mus) /
                                         (np.expand_dims(self.hyper_prior.kappa + self.prior.kappas, axis=(1, 2))), axis=0) / self.size
                                + np.sum(xxTk, axis=0) / self.size - np.einsum('kd,kl->dl', self.posterior.mus, xk) / self.size
                                - np.einsum('kd,kl->dl', xk, self.posterior.mus) / self.size
                                + np.einsum('k,kd,kl->dl', nk, self.posterior.mus, self.posterior.mus) / self.size)
            nu = np.sum(self.hyper_prior.wishart.nu + nk + 1) / self.size

            self.hyper_posterior.params = mu, kappa, psi, nu

        _, lmbda = self.hyper_posterior.mode()
        mus = self.posterior.mode()
        self.likelihood.mus = mus
        self.likelihood.lmbdas = np.stack(self.size * [lmbda])

    def expected_log_likelihood(self, x):
        from scipy.special import digamma

        E_mu_lmbda = np.einsum('kd,dl->kl', self.posterior.mus, self.hyper_posterior.wishart.nu * self.hyper_posterior.wishart.psi)
        E_mu_lmbda_muT = - 0.5 * np.einsum('kd,kd->k', E_mu_lmbda, self.posterior.mus)
        E_lmbda = - 0.5 * (self.hyper_posterior.wishart.nu * self.hyper_posterior.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.hyper_posterior.wishart.nu - np.arange(self.dim)) / 2.))
                                + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.hyper_posterior.wishart.psi_chol))))

        xk, nk, xxTk, nk = self.likelihood.statistics(x, fold=False)
        xxTk += np.expand_dims(np.linalg.inv(self.posterior.omegas), axis=1)

        log_base = self.likelihood.log_base()

        return np.expand_dims(log_base, axis=1)\
               + np.einsum('kd,knd->kn', E_mu_lmbda, xk)\
               + np.einsum('k,kn->kn', E_mu_lmbda_muT, nk)\
               + np.einsum('kdl,kndl->kn', np.stack(self.size * [E_lmbda]), xxTk)\
               + np.einsum('k,kn->kn', np.expand_dims(E_logdet_lmbda, axis=0), nk)

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


class LinearGaussianWithMatrixNormalWishart:
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal Wishart prior.
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
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, y, x, weights=None):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.mode()

    # Gibbs sampling
    def resample(self, y=[], x=[]):
        stats = self.likelihood.statistics(y, x)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    # Mean field
    def meanfield_update(self, y, x, weights=None):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.likelihood.params = self.posterior.rvs()

    def meanfield_sgdstep(self, y, x, weights, prob, stepsize):
        stats = self.likelihood.statistics(y, x) if weights is None\
            else self.likelihood.weighted_statistics(y, x, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.likelihood.params = self.posterior.rvs()

    def variational_lowerbound(self):
        q_entropy = self.posterior.entropy()
        qp_cross_entropy = self.posterior.cross_entropy(self.prior)
        return q_entropy - qp_cross_entropy

    def posterior_predictive_gaussian(self, x, aleatoric_only=False):
        x = np.reshape(x, (-1, self.likelihood.dcol))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.drow + 1
        mus = np.einsum('kh,...h->...k', M, x)

        if aleatoric_only:
            lmbdas = np.tile(psi * df, (len(x), 1, 1))
        else:
            c = 1. + np.einsum('...k,...kh,...h->...', x, np.linalg.inv(K), x)
            lmbdas = np.einsum('kh,...->...kh', psi, df / c)
        return mus, lmbdas

    def log_posterior_predictive_gaussian(self, y, x):
        mus, lmbdas = self.posterior_predictive_gaussian(x)
        return mvn_logpdf(y, mus, lmbdas)

    def posterior_predictive_studentt(self, x, aleatoric_only=False):
        x = np.reshape(x, (-1, self.likelihood.dcol))

        if self.likelihood.affine:
            x = np.hstack((x, np.ones((len(x), 1))))

        M, K, psi, nu = self.posterior.params

        df = nu - self.likelihood.drow + 1
        mus = np.einsum('kh,...h->...k', M, x)

        if aleatoric_only:
            lmbdas = np.tile(psi * df, (len(x), 1, 1))
        else:
            c = 1. + np.einsum('...k,...kh,...h->...', x, np.linalg.inv(K), x)
            lmbdas = np.einsum('kh,...->...kh', psi, df / c)
        return mus, lmbdas, df

    def log_posterior_predictive_studentt(self, y, x):
        mus, lmbdas, df = self.posterior_predictive_studentt(x)
        return mvt_logpdf(y, mu=mus, lmbda=lmbdas, df=df)
