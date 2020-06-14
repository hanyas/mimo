import copy
from abc import ABC

import numpy as np
from numpy.core._umath_tests import inner1d

from scipy.special import digamma
from scipy.special._ufuncs import gammaln, betaln

from mimo.abstraction import Statistics as Stats

from mimo.distributions import Categorical
from mimo.distributions import Gaussian
from mimo.distributions import GaussianWithFixedCovariance
from mimo.distributions import DiagonalGaussian
from mimo.distributions import TiedGaussians
from mimo.distributions import LinearGaussian
from mimo.distributions import NormalInverseWishart

from mimo.util.matrix import blockarray


class CategoricalWithDirichlet(Categorical, ABC):
    """
    This class is a categorical distribution over labels
     with a Dirichlet distribution as prior.
    Parameters:
        probs, a vector encoding a finite pmf
    """

    def __init__(self, prior,  K=None, probs=None):
        super(CategoricalWithDirichlet, self).__init__(K=K, probs=probs)

        # Dirichlet prior
        self.prior = prior

        # Dirichlet posterior
        self.posterior = copy.deepcopy(prior)

        self.K, self.probs = K, probs
        if K is None or probs is None:
            self.probs = self.prior.rvs()
            self.K = prior.K

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        counts = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.alphas = self.prior.alphas + counts

        self.params = self.posterior.mean()  # mode might be undefined
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        self.posterior.alphas = self.prior.alphas\
                                + self.get_statistics(data)

        _probs = self.posterior.rvs()
        self.probs = np.clip(_probs, np.spacing(1.), np.inf)
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        counts = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.alphas = self.prior.alphas + counts

        self.probs = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.get_weighted_statistics(data, weights)
        self.posterior.alphas = (1. - stepsize) * self.posterior.alphas\
                                   + stepsize * (self.prior.alphas + 1. / prob * stats)

        self.probs = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood(np.arange(self.K))

        q_entropy = -1. * (gammaln(self.posterior.alphas.sum())
                           - gammaln(self.posterior.alphas).sum()
                           + ((self.posterior.alphas - 1.) * logpitilde).sum())

        p_avgengy = gammaln(self.prior.alphas.sum())\
                    - gammaln(self.prior.alphas).sum()\
                    + ((self.prior.alphas - 1.) * logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self, x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else np.arange(self.K)
        return digamma(self.posterior.alphas[x]) - digamma(self.posterior.alphas.sum())


class CategoricalWithStickBreaking(Categorical, ABC):
    """
    This class is a categorical distribution over labels
     with a stick-breaking process as prior.
    Parameters:
        probs, a vector encoding a finite pmf
    """

    def __init__(self, prior, K=None, probs=None):
        super(CategoricalWithStickBreaking, self).__init__(K=K, probs=probs)

        # stick-breaking prior
        self.prior = prior

        # stick-breaking posterior
        self.posterior = copy.deepcopy(prior)

        self.K, self.probs = K, probs
        if K is None or probs is None:
            self.probs = self.prior.rvs()
            self.K = prior.K

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        counts = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))
        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.params = self.posterior.mean()  # mode might be undefined
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        counts = self.get_statistics(data)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))
        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.probs = self.posterior.rvs()
        return self

    # Mean field
    def meanfield_update(self, data, weights=None):
        counts = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.probs = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        counts = self.get_weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = (1. - stepsize) * self.posterior.gammas\
                                + stepsize * (self.prior.gammas + 1. / prob * counts)
        self.posterior.deltas = (1. - stepsize) * self.posterior.deltas\
                                + stepsize * (self.prior.deltas + 1. / prob * cumcounts)

        self.probs = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        # entropy of a beta distribution https://en.wikipedia.org/wiki/Beta_distribution
        # E_q[log(q(pi))] = entropy of beta distribution of variational posterior
        q_entropy = np.sum(betaln(self.posterior.gammas, self.posterior.deltas)
                           - (self.posterior.gammas - 1.) * digamma(self.posterior.gammas)
                           - (self.posterior.deltas - 1.) * digamma(self.posterior.deltas)
                           + (self.posterior.gammas + self.posterior.deltas - 2.)
                           * digamma(self.posterior.gammas + self.posterior.deltas))

        # cross entropy of a beta distribution https://en.wikipedia.org/wiki/Beta_distribution
        # E_q[log(p(pi))] = cross entropy of beta dists and the stick-breaking prior
        p_avgengy = -1.0 * np.sum(betaln(self.prior.gammas, self.prior.deltas)
                                  - (self.prior.gammas - 1.) * digamma(self.posterior.deltas)
                                  + (self.prior.gammas - 1.) * digamma(self.posterior.gammas + self.posterior.deltas))

        return p_avgengy + q_entropy

    def expected_log_likelihood(self, x):
        E_log_stick = digamma(self.posterior.gammas)\
                      - digamma(self.posterior.gammas + self.posterior.deltas)

        E_log_rest = digamma(self.posterior.deltas)\
                     - digamma(self.posterior.gammas + self.posterior.deltas)
        return E_log_stick, E_log_rest


class GaussianWithNormalInverseWishart(Gaussian, ABC):
    """
    Multivariate Gaussian distribution class.
    Uses a Normal-Inverse-Wishart prior and posterior
    Parameters are mean and covariance matrix:
        mu, sigma
    """

    def __init__(self, prior, mu=None, sigma=None):
        super(GaussianWithNormalInverseWishart, self).__init__(mu=mu, sigma=sigma)
        # Normal-Inverse Wishart conjugate
        self.prior = prior

        # Normal-Inverse Wishart posterior
        self.posterior = copy.deepcopy(prior)

        self.mu, self.sigma = mu, sigma
        if mu is None or sigma is None:
            self.mu, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        self.prior.nat_param = self.get_statistics(data)
        self.mu, self.sigma = self.prior.rvs()
        return self

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        stats = self.get_statistics(data)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.params = self.posterior.rvs()
        return self

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        return np.sum(digamma((self.posterior.invwishart.nu - np.arange(self.dim)) / 2.)) \
               + self.dim * np.log(2) - 2. * np.sum(np.log(np.diag(self.posterior.invwishart.psi_chol)))

    def variational_lowerbound(self):
        loglmbdatilde = self._loglmbdatilde()

        # see Eq. 10.77 in Bishop
        q_entropy = - 1. * (0.5 * (loglmbdatilde + self.dim * (np.log(self.posterior.kappa / (2 * np.pi)) - 1.))
                            - self.posterior.invwishart.entropy())

        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (self.dim * np.log(self.prior.kappa / (2. * np.pi)) + loglmbdatilde
                           - self.dim * self.prior.kappa / self.posterior.kappa
                           - self.prior.kappa * self.posterior.invwishart.nu
                           * np.dot(self.posterior.gaussian.mu - self.prior.gaussian.mu,
                                    np.linalg.solve(self.posterior.invwishart.psi,
                                                    self.posterior.gaussian.mu - self.prior.gaussian.mu))) \
                    - self.prior.invwishart.log_partition() \
                    + (self.prior.invwishart.nu - self.dim - 1.) / 2. * loglmbdatilde - 0.5 * self.posterior.invwishart.nu \
                    * np.linalg.solve(self.posterior.invwishart.psi, self.prior.invwishart.psi).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self, x):
        x = np.reshape(x, (-1, self.dim)) - self.posterior.gaussian.mu
        xs = np.linalg.solve(self.posterior.invwishart.psi_chol, x.T)

        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        # sneaky gaussian/quadratic identity hidden here
        return 0.5 * self._loglmbdatilde() - self.dim / (2. * self.posterior.kappa)\
               - self.posterior.invwishart.nu / 2. * inner1d(xs.T, xs.T) \
               - self.dim / 2. * np.log(2. * np.pi)

    def log_marginal_likelihood(self, x):
        x = np.atleast_2d(x)

        stats = self.prior.get_statistics(x)
        natparam = self.prior.nat_param + stats
        params = NormalInverseWishart.nat_to_std(natparam)

        log_partition_prior = self.prior.log_partition()
        log_partition_posterior = self.posterior.log_partition(params)

        return log_partition_posterior - log_partition_prior \
               - 0.5 * len(x) * self.dim * np.log(np.pi)

    def log_posterior_predictive_gaussian(self, x):
        x = np.atleast_2d(x)

        stats = self.get_statistics(x)
        natparam = self.posterior.nat_param + stats
        mu, kappa, psi, nu = NormalInverseWishart.nat_to_std(natparam)

        loc = mu
        scale = psi / nu

        from mimo.util.stats import multivariate_gaussian_loglik
        return multivariate_gaussian_loglik(x, loc, scale)

    def log_posterior_predictive_studentt(self, x):
        x = np.atleast_2d(x)

        stats = self.get_statistics(x)
        natparam = self.posterior.nat_param + stats
        mu, kappa, psi, nu = NormalInverseWishart.nat_to_std(natparam)

        loc = mu
        df = nu - self.dim + 1
        scale = (kappa + 1) / (kappa * (nu - self.dim + 1)) * psi

        from mimo.util.stats import multivariate_studentt_loglik
        return multivariate_studentt_loglik(x, loc, scale, df)


class GaussianWithNormal(GaussianWithFixedCovariance, ABC):
    """
    Multivariate Gaussian distribution class.
    Uses a Gaussian prior and posterior over the mean
    Parameters are mean and covariance matrix:
        mu, sigma
    """

    def __init__(self, prior, sigma, mu=None):
        assert sigma is not None

        super(GaussianWithNormal, self).__init__(mu=mu, sigma=sigma)
        # Normal prior
        self.prior = prior

        # Normal posterior
        self.posterior = copy.deepcopy(prior)

        self.mu = mu if mu is not None else self.prior.rvs()

    def empirical_bayes(self, data):
        self.prior.nat_param = self.get_statistics(data)
        self.mu = self.prior.rvs()
        return self

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        x, n, _, _ = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)

        _sigma = self.sigma_inv
        _mu = _sigma @ (x / n)
        self.posterior.nat_param = Stats([_mu, _sigma]) + self.prior.nat_param

        self.mu = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        x, n, _, _ = self.get_statistics(data)

        _sigma = self.sigma_inv
        _mu = _sigma @ (x / n)
        self.posterior.nat_param = Stats([_mu, _sigma]) + self.prior.nat_param

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        return new


class GaussianWithNormalInverseGamma(DiagonalGaussian, ABC):
    """
    Multivariate Diagonal Gaussian distribution class.
    Uses a Normal-Inverse-Gamma prior and posterior
    Parameters are mean and covariance matrix:
        mu, sigmas
    """

    def __init__(self, prior, mu=None, sigmas=None):
        super(GaussianWithNormalInverseGamma, self).__init__(mu=mu, sigmas=sigmas)
        # Normal-Inverse Wishart conjugate
        self.prior = prior

        # Normal-Inverse Wishart posterior
        self.posterior = copy.deepcopy(prior)

        if mu is None or sigmas is None:
            self.mu, self.sigma = self.prior.rvs()
        else:
            self.mu, self.sigma = mu, sigmas

    def empirical_bayes(self, data):
        self.prior.nat_param = self.get_statistics(data)
        self.mu, self.sigma = self.prior.rvs()
        return self

    # Max a posteriori
    def max_aposteriori(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        self.posterior.nat_param = self.prior.nat_param\
                                   + self.get_statistics(data)

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfield_update(self, data, weights=None):
        stats = self.get_statistics(data) if weights is None\
            else self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        stats = self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param + 1. / prob * stats)

        self.params = self.posterior.rvs()
        return self

    def variational_lowerbound(self):
        expected_stats = self.posterior.get_expected_statistics()

        param_diff = self.prior.nat_param - self.posterior.nat_param
        aux = sum(v.dot(w) for v, w in zip(param_diff, expected_stats))

        logpart_diff = self.prior.log_partition() - self.posterior.log_partition()

        return aux - logpart_diff

    def expected_log_likelihood(self, x):
        a, b, c, d = self.posterior.get_expected_statistics()
        return (x**2).dot(a) + x.dot(b) + c.sum() + d.sum()\
               - 0.5 * self.dim * np.log(2. * np.pi)


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
        stats = self.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.mode()
        return self

    # Gibbs sampling
    def resample(self, data=[], labels=[]):
        stats = self.get_statistics(data, labels)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.params = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mus = self.mus.copy()
        new.sigma = self.sigma.copy()
        return new


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
        stats = self.posterior.get_statistics(y, x) if weights is None\
            else self.posterior.get_weighted_statistics(y, x, weights)
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
        stats = self.get_weighted_statistics(y, x, weights)
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