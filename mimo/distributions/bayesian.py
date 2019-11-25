import numpy as np

import copy

from mimo.abstractions import MaxLikelihood, MAP
from mimo.abstractions import GibbsSampling, MeanField, MeanFieldSVI

from mimo.distributions.gaussian import Gaussian, DiagonalGaussian
from mimo.distributions.categorical import Categorical
from mimo.distributions import LinearGaussian, LinearGaussianWithNoisyInputs

from scipy.special import digamma, gammaln, betaln
from numpy.core.umath_tests import inner1d
from mimo.util.general import blockarray


class BayesianGaussian(Gaussian, MaxLikelihood, MAP,
                       GibbsSampling, MeanField, MeanFieldSVI):
    """
    Multivariate Gaussian distribution class.
    Uses a Normal-Inverse-Wishart prior and posterior
    Parameters are mean and covariance matrix:
        mu, sigma
    """

    def __init__(self, prior, mu=None, sigma=None):
        super(BayesianGaussian, self).__init__(mu=mu, sigma=sigma)
        # Normal-Inverse Wishart conjugate
        self.prior = prior

        # Normal-Inverse Wishart posterior
        self.posterior = copy.deepcopy(prior)

        if mu is None or sigma is None:
            self.mu, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        # initialize prior from data
        self.prior.nat_param = self.prior.get_statistics(data)
        # intialize from prior given new hyperparameters
        self.mu, self.sigma = self.prior.rvs()
        return self

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)

        x, n, xxT, n = stats
        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n < self.dim or np.sum(np.linalg.svd(xxT, compute_uv=False) > 1e-6) < self.dim:
            self.mu = 99999999 * np.ones(self.dim)
            self.sigma = np.eye(self.dim)
        else:
            self.mu = x / n
            self.sigma = xxT / n - np.outer(self.mu, self.mu)
        return self

    # Max a posteriori
    def MAP(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        stats += self.prior.nat_param

        x, n, xxT, n = stats
        self.mu = x / n
        self.sigma = xxT / n - np.outer(self.mu, self.mu)
        return self

    # Gibbs sampling
    def resample(self, data=[], importance=[]):
        # importance imply importance of samples, i.e. exponentiated likelihood
        if not importance:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_statistics(data)
        else:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_weighted_statistics(data, weights=importance)

        self.mu, self.sigma = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.mu, self.sigma = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param
                                               + 1. / prob * self.posterior.get_weighted_statistics(data, weights))

        self.mu, self.sigma = self.posterior.rvs()
        return self

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        return np.sum(digamma((self.posterior.invwishart.nu - np.arange(self.dim)) / 2.)) \
               + self.dim * np.log(2) - 2. * np.sum(np.log(np.diag(self.posterior.invwishart.psi_chol)))

    def get_vlb(self):
        loglmbdatilde = self._loglmbdatilde()

        # see Eq. 10.77 in Bishop
        q_entropy = - 1. * (0.5 * (loglmbdatilde + self.dim * (np.log(self.posterior.kappa / (2 * np.pi)) - 1.))
                            - self.posterior.invwishart.entropy())

        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (self.dim * np.log(self.prior.kappa / (2. * np.pi)) + loglmbdatilde
                           - self.dim * self.prior.kappa / self.posterior.kappa - self.prior.kappa * self.posterior.invwishart.nu
                           * np.dot(self.posterior.gaussian.mu - self.prior.gaussian.mu, np.linalg.solve(self.posterior.invwishart.psi, self.posterior.gaussian.mu - self.prior.gaussian.mu))) \
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
               - self.posterior.invwishart.nu / 2. * inner1d(xs.T, xs.T) - self.dim / 2. * np.log(2. * np.pi)


class BayesianDiagonalGaussian(DiagonalGaussian, MaxLikelihood, MAP,
                               GibbsSampling, MeanField, MeanFieldSVI):
    """
    Product of normal-gamma priors over mu (mean vector) and sigmas
    (vector of scalar variances).
    Uses a Normal-Inverse-Gamma prior and posterior
    It allows placing different prior hyperparameters on different components.
    """

    def __init__(self, prior, mu=None, sigmas=None):
        super(BayesianDiagonalGaussian, self).__init__(mu=mu, sigmas=sigmas)
        # Normal-Inverse Wishart conjugate
        self.prior = prior

        # Normal-Inverse Gamma posterior
        self.posterior = copy.deepcopy(prior)

        if mu is None or sigmas is None:
            self.mu, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        # initialize prior from data
        self.prior.nat_param = self.prior.get_statistics(data)
        # intialize from prior given new hyperparameters
        self.mu, self.sigma = self.prior.rvs()
        return self

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)

        x, n, n, xx = stats

        self.mu = x / n
        self.sigma = xx / n - self.mu**2

        return self

    # Max a posteriori
    def MAP(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        stats += self.prior.nat_param

        x, n, n, xx = stats

        self.mu = x / n
        self.sigma = xx / n - self.mu**2

        return self

    # Gibbs sampling
    def resample(self, data=[], importance=[]):
        # importance imply importance of samples, i.e. exponentiated likelihood
        if not importance:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_statistics(data)
        else:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_weighted_statistics(data, weights=importance)

        self.mu, self.sigma = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigmas = self.sigmas.copy()
        return new

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        # for plotting
        self.mu, self.sigma = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param
                                                 + 1. / prob * self.posterior.get_weighted_statistics(data, weights))

        # for plotting
        self.mu, self.sigma = self.posterior.rvs()
        return self

    def get_vlb(self):
        expected_stats = self.posterior.get_expected_statistics()

        param_diff = self.prior.nat_param - self.posterior.nat_param
        aux = sum(v.dot(w) for v, w in zip(param_diff, expected_stats))

        logpart_diff = self.prior.log_partition() - self.posterior.log_partition()

        return aux - logpart_diff

    def expected_log_likelihood(self, x, stats=None):
        a, b, c, d = self.posterior.get_expected_statistics()
        return (x**2).dot(a) + x.dot(b) + c.sum() + d.sum() - 0.5 * self.D * np.log(2. * np.pi)


class BayesianCategoricalWithDirichlet(Categorical,  MaxLikelihood, MAP,
                                       GibbsSampling, MeanField, MeanFieldSVI):
    """
    This class represents a categorical distribution over labels, where the
    parameter is weights and the prior is a Dirichlet distribution.
    For example, if K == 3, then five samples may look like [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls. In other
    words, generated data or data passed to log_likelihood are indices, not
    indicator variables!  (But when 'weighted data' is passed, like in mean
    field or weighted max likelihood, the weights are over indicator
    variables...)
    This class can be used as a weak limit approximation for a DP, particularly by
    calling __init__ with alpha_0 and K arguments, in which case the prior will be
    a symmetric Dirichlet with K components and parameter alpha_0/K; K is then the
    weak limit approximation parameter.
    Parameters:
        probs, a vector encoding a finite pmf
    """

    def __init__(self, prior,  K=None, probs=None):
        super(BayesianCategoricalWithDirichlet, self).__init__(K=K, probs=probs)

        # Dirichlet prior
        self.prior = prior

        # Dirichlet posterior
        self.posterior = copy.deepcopy(prior)

        if K is None or probs is None:
            self.probs = self.prior.rvs()
            self.K = prior.K

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        if weights is None:
            counts = self.posterior.get_statistics(data)
        else:
            counts = self.posterior.get_weighted_statistics(data, weights)
        self.probs = counts / counts.sum()
        return self

    def MAP(self, data, weights=None):
        if weights is None:
            counts = self.posterior.get_statistics(data)
        else:
            counts = self.posterior.get_weighted_statistics(data, weights)
        counts += self.prior.alphas
        self.probs = counts / counts.sum()
        return self

    # Gibbs sampling
    def resample(self, data=[]):
        self.posterior.alphas = self.prior.alphas + self.posterior.get_statistics(data)
        _probs = self.posterior.rvs()
        self.probs = np.clip(_probs, np.spacing(1.), np.inf)
        return self

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            counts = self.posterior.get_statistics(data)
        else:
            counts = self.posterior.get_weighted_statistics(data, weights)
        self.posterior.alphas = self.prior.alphas + counts

        self.probs = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        self.posterior.alphas = (1. - stepsize) * self.posterior.alphas\
                                + stepsize * (self.prior.alphas
                                              + 1. / prob * self.posterior.get_weighted_statistics(data, weights))

        self.probs = self.posterior.rvs()
        return self

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood()  # default is on np.arange(self.K)

        q_entropy = -1. * (gammaln(self.posterior.alphas.sum())
                           - gammaln(self.posterior.alphas).sum()
                           + ((self.posterior.alphas - 1.) * logpitilde).sum())

        p_avgengy = gammaln(self.prior.alphas.sum())\
                    - gammaln(self.prior.alphas).sum()\
                    + ((self.prior.alphas - 1.) * logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self, x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else slice(None)
        return digamma(self.posterior.alphas[x]) - digamma(self.posterior.alphas.sum())


class BayesianCategoricalWithStickBreaking(Categorical, GibbsSampling, MeanField, MeanFieldSVI):
    """
    This class represents a categorical distribution over labels, where the
    parameter is weights and the prior is a stick-breaking process.
    For example, if K == 3, then five samples may look like [0,1,0,2,1]
    This class can be seen as a better approximation for a DP via a stick-breaking prior
    Parameters:
        weights, vector encoding a finite pmf
    """

    def __init__(self, prior, K=None, probs=None):
        super(BayesianCategoricalWithStickBreaking, self).__init__(K=K, probs=probs)

        # stick-breaking prior
        self.prior = prior

        # stick-breaking posterior
        self.posterior = copy.deepcopy(prior)

        if K is None or probs is None:
            self.probs = self.prior.rvs()
            self.K = prior.K

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, data=[]):
        counts = self.posterior.get_statistics(data)
        # see Blei et. al Variational Inference for Dirichlet Process Mixtures
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))
        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.probs = self.posterior.rvs()
        return self

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            counts = self.posterior.get_statistics(data)
        else:
            counts = self.posterior.get_weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = self.prior.gammas + counts
        self.posterior.deltas = self.prior.deltas + cumcounts

        self.probs = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        counts = self.posterior.get_weighted_statistics(data, weights)
        cumcounts = np.hstack((np.cumsum(counts[::-1])[-2::-1], 0))

        self.posterior.gammas = (1. - stepsize) * self.posterior.gammas\
                                + stepsize * (self.prior.gammas + 1. / prob * counts)
        self.posterior.deltas = (1. - stepsize) * self.posterior.deltas\
                                + stepsize * (self.prior.deltas + 1. / prob * cumcounts)

        self.probs = self.posterior.rvs()
        return self

    def get_vlb(self):
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

    def expected_log_likelihood(self, x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else slice(None)
        return np.take(digamma(self.posterior.gammas) - digamma(self.posterior.gammas + self.posterior.deltas)
                       + np.hstack((0, np.cumsum(digamma(self.posterior.deltas)
                                                 - digamma(self.posterior.gammas + self.posterior.deltas))[:-1])), x)


class BayesianLinearGaussian(LinearGaussian, MaxLikelihood,
                             GibbsSampling, MeanField, MeanFieldSVI):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal/Inverse-Wishart prior.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """

    def __init__(self, prior, A=None, sigma=None):
        super(BayesianLinearGaussian, self).__init__(A=A, sigma=sigma, affine=prior.affine)

        self.A = A
        self.sigma = sigma

        # Matrix-Normal-Inv-Wishart prior
        self.prior = prior

        # Matrix-Normal-Inv-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        if A is None or sigma is None:
            self.A, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)

        # (yxT, xxT, yyT, n)
        yxT, xxT, yyT, n = stats

        self.A = np.linalg.solve(xxT, yxT.T).T
        self.sigma = (yyT - self.A.dot(yxT.T)) / n

        def symmetrize(A):
            return (A + A.T) / 2.

        # numerical stabilization
        self.sigma = 1e-10 * np.eye(self.dout) + symmetrize(self.sigma)

        assert np.allclose(self.sigma, self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        return self

    # Max a posteriori
    def MAP(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        stats += self.prior.nat_param

        # (yxT, xxT, yyT, n)
        yxT, xxT, yyT, n = stats

        self.A = np.linalg.solve(xxT, yxT.T).T
        self.sigma = (yyT - self.A.dot(yxT.T)) / n

        def symmetrize(A):
            return (A + A.T) / 2.

        # numerical stabilization
        self.sigma = 1e-10 * np.eye(self.dout) + symmetrize(self.sigma)

        assert np.allclose(self.sigma, self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        return self

    # Gibbs sampling
    def resample(self, data=[], importance=[]):
        # importance imply importance of samples, i.e. exponentiated likelihood
        if not importance:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_statistics(data)
        else:
            self.posterior.nat_param = self.prior.nat_param\
                                       + self.posterior.get_weighted_statistics(data, weights=importance)

        self.A, self.sigma = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.A = self.A.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        self.posterior.nat_param = self.prior.nat_param + stats

        self.A, self.sigma = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param\
                                   + stepsize * (self.prior.nat_param
                                                 + 1. / prob * self.posterior.get_weighted_statistics(data, weights))

        self.A, self.sigma = self.posterior.rvs()
        return self

    def get_vlb(self):
        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = self.posterior.get_expected_statistics()
        a, b, c, d = self.prior.nat_param - self.posterior.nat_param

        aux = - 0.5 * np.trace(c.dot(E_Sigmainv)) + np.trace(a.T.dot(E_Sigmainv_A))\
              - 0.5 * np.trace(b.dot(E_AT_Sigmainv_A)) + 0.5 * d * E_logdetSigmainv

        logpart_diff = self.prior.log_partition() - self.posterior.log_partition()

        return aux - logpart_diff

    def expected_log_likelihood(self, xy=None, stats=None):
        assert isinstance(xy, (tuple, np.ndarray)) ^ isinstance(stats, tuple)

        dout = self.dout
        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = self.posterior.get_expected_statistics()

        if self.affine:
            E_Sigmainv_A, E_Sigmainv_b = E_Sigmainv_A[:, :-1], E_Sigmainv_A[:, -1]
            E_AT_Sigmainv_A, E_AT_Sigmainv_b, E_bT_Sigmainv_b = E_AT_Sigmainv_A[:-1, :-1], E_AT_Sigmainv_A[:-1, -1], E_AT_Sigmainv_A[-1, -1]

        x, y = (xy[:, :-dout], xy[:, -dout:]) if isinstance(xy, np.ndarray) else xy

        parammat = -1. / 2 * blockarray([[E_AT_Sigmainv_A, -E_Sigmainv_A.T], [-E_Sigmainv_A, E_Sigmainv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum('ni,ni->n', xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-dout, :-dout]), x)
            out += np.einsum(contract, y.dot(parammat[-dout:, -dout:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-dout, -dout:]), y)

        out += - dout / 2. * np.log(2 * np.pi) + 1. / 2 * E_logdetSigmainv

        if self.affine:
            out += y.dot(E_Sigmainv_b)
            out -= x.dot(E_AT_Sigmainv_b)
            out -= 1. / 2 * E_bT_Sigmainv_b

        return out


class BayesianLinearGaussianWithNoisyInputs(LinearGaussianWithNoisyInputs, MaxLikelihood,
                                            GibbsSampling, MeanField, MeanFieldSVI):
    """
    Multivariate Gaussian distribution with a linear mean function.
    Uses a conjugate Matrix-Normal/Inverse-Wishart prior.
    Parameters are linear transf. and covariance matrix:
        A, sigma
    """

    def __init__(self, prior, mu=None, sigma_niw=None, A=None, sigma=None):
        super(BayesianLinearGaussianWithNoisyInputs, self).__init__(mu=mu, sigma_niw=sigma_niw,
                                                                    A=A, sigma=sigma, affine=prior.affine)

        self.mu = mu
        self.sigma_niw = sigma_niw

        self.A = A
        self.sigma = sigma

        # Normal-Inverse-Wishart-Matrix-Normal-Inverse-Wishart prior
        self.prior = prior

        # Normal-Inverse-Wishart-Matrix-Normal-Inverse-Wishart posterior
        self.posterior = copy.deepcopy(prior)

        if A is None or sigma is None:
            self.mu, self.sigma_niw, self.A, self.sigma = self.prior.rvs()

    # def __init__(self, prior, mu=None, sigma=None):
    #     super(BayesianGaussian, self).__init__(mu=mu, sigma=sigma)
    #     # Normal-Inverse Wishart conjugate
    #     self.prior = prior
    #
    #     # Normal-Inverse Wishart posterior
    #     self.posterior = copy.deepcopy(prior)
    #
    #     if mu is None or sigma is None:
    #         self.mu, self.sigma = self.prior.rvs()

    def empirical_bayes(self, data):
        raise NotImplementedError

    # Max likelihood
    def max_likelihood(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)

        # (yxT, xxT, yyT, n)
        x_niw, n_niw, xxT_niw, n_niw, yxT_mniw, xxT_mniw, yyT_mniw, n_mniw = stats

        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n_niw < self.dim or np.sum(np.linalg.svd(xxT_niw, compute_uv=False) > 1e-6) < self.dim:
            self.mu = 99999999 * np.ones(self.dim)
            self.sigma_niw = np.eye(self.dim)
        else:
            self.mu = x_niw / n_niw
            self.sigma_niw = xxT_niw / n_niw - np.outer(self.mu, self.mu)

        self.A = np.linalg.solve(xxT_mniw, yxT_mniw.T).T
        self.sigma = (yyT_mniw - self.A.dot(yxT_mniw.T)) / n_mniw

        def symmetrize(A):
            return (A + A.T) / 2.

        # numerical stabilization
        self.sigma = 1e-10 * np.eye(self.dout) + symmetrize(self.sigma)

        assert np.allclose(self.sigma, self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        return self

    # # Max likelihood
    # def max_likelihood(self, data, weights=None):
    #     if weights is None:
    #         stats = self.posterior.get_statistics(data)
    #     else:
    #         stats = self.posterior.get_weighted_statistics(data, weights)
    #
    #     x, n, xxT, n = stats
    #     # this SVD is necessary to check if the max likelihood solution is
    #     # degenerate, which can happen in the EM algorithm
    #     if n < self.dim or np.sum(np.linalg.svd(xxT, compute_uv=False) > 1e-6) < self.dim:
    #         self.mu = 99999999 * np.ones(self.dim)
    #         self.sigma = np.eye(self.dim)
    #     else:
    #         self.mu = x / n
    #         self.sigma = xxT / n - np.outer(self.mu, self.mu)
    #     return self

    # Max a posteriori
    def MAP(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)
        stats += self.prior.nat_param

        # (yxT, xxT, yyT, n)
        x_niw, n_niw, xxT_niw, n_niw, yxT_mniw, xxT_mniw, yyT_mniw, n_mniw = stats

        self.mu = x_niw / n_niw
        self.sigma_niw = xxT_niw / n_niw - np.outer(self.mu, self.mu)

        self.A = np.linalg.solve(xxT_mniw, yxT_mniw.T).T
        self.sigma = (yyT_mniw - self.A.dot(yxT_mniw.T)) / n_mniw

        def symmetrize(A):
            return (A + A.T) / 2.

        # numerical stabilization
        self.sigma = 1e-10 * np.eye(self.dout) + symmetrize(self.sigma)

        assert np.allclose(self.sigma, self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        return self

    # # Max a posteriori
    # def MAP(self, data, weights=None):
    #     if weights is None:
    #         stats = self.posterior.get_statistics(data)
    #     else:
    #         stats = self.posterior.get_weighted_statistics(data, weights)
    #     stats += self.prior.nat_param
    #
    #     x, n, xxT, n = stats
    #     self.mu = x / n
    #     self.sigma = xxT / n - np.outer(self.mu, self.mu)
    #     return self

    # Gibbs sampling
    def resample(self, data=[], importance=[]):
        # importance imply importance of samples, i.e. exponentiated likelihood
        if not importance:
            self.posterior.nat_param = self.prior.nat_param +\
                                       self.posterior.get_statistics(data)
        else:
            self.posterior.nat_param = self.prior.nat_param +\
                                       self.posterior.get_weighted_statistics(data, weights=importance)

        self.mu, self.sigma_niw, self.A, self.sigma = self.posterior.rvs()
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma_niw = self.sigma_niw.copy()
        new.A = self.A.copy()
        new.sigma = self.sigma.copy()
        return new

    # Mean field
    def meanfieldupdate(self, data, weights=None):
        if weights is None:
            stats = self.posterior.get_statistics(data)
        else:
            stats = self.posterior.get_weighted_statistics(data, weights)

        self.posterior.nat_param = self.prior.nat_param + stats
        self.mu, self.sigma_niw, self.A, self.sigma = self.posterior.rvs()
        return self

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        self.posterior.nat_param = (1. - stepsize) * self.posterior.nat_param +\
                                   stepsize * (self.prior.nat_param +
                                               1. / prob * self.posterior.get_weighted_statistics(data, weights))

        self.mu, self.sigma_niw, self.A, self.sigma = self.posterior.rvs()
        return self

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        return np.sum(digamma((self.posterior.invwishart_niw.nu - np.arange(self.dim)) / 2.)) \
               + self.dim * np.log(2) - 2. * np.sum(np.log(np.diag(self.posterior.invwishart_niw.psi_chol)))

    def get_vlb(self):
        trash1, trash2, trash3, trash4, E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = self.posterior.get_expected_statistics()
        trash5, trash6, trash7, trash8, a, b, c, d = self.prior.nat_param - self.posterior.nat_param

        aux = - 0.5 * np.trace(c.dot(E_Sigmainv)) + np.trace(a.T.dot(E_Sigmainv_A)) -\
              0.5 * np.trace(b.dot(E_AT_Sigmainv_A)) + 0.5 * d * E_logdetSigmainv

        logpart_diff = self.prior.log_partition() - self.posterior.log_partition()

        loglmbdatilde = self._loglmbdatilde()

        # see Eq. 10.77 in Bishop
        q_entropy = - 1. * (0.5 * (loglmbdatilde + self.dim * (np.log(self.posterior.kappa / (2 * np.pi)) - 1.)) -
                            self.posterior.invwishart_niw.entropy())

        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (self.dim * np.log(self.prior.kappa / (2. * np.pi)) + loglmbdatilde
                           - self.dim * self.prior.kappa / self.posterior.kappa - self.prior.kappa * self.posterior.invwishart_niw.nu
                           * np.dot(self.posterior.gaussian.mu - self.prior.gaussian.mu,
                                    np.linalg.solve(self.posterior.invwishart_niw.psi,
                                                    self.posterior.gaussian.mu - self.prior.gaussian.mu))) \
                    - self.prior.invwishart_niw.log_partition() \
                    + (
                            self.prior.invwishart_niw.nu - self.dim - 1.) / 2. * loglmbdatilde - 0.5 * self.posterior.invwishart_niw.nu \
                    * np.linalg.solve(self.posterior.invwishart_niw.psi, self.prior.invwishart_niw.psi).trace()

        return aux - logpart_diff + q_entropy + p_avgengy

    # def get_vlb(self):
    #     loglmbdatilde = self._loglmbdatilde()
    #
    #     # see Eq. 10.77 in Bishop
    #     q_entropy = - 1. * (0.5 * (loglmbdatilde + self.dim * (np.log(self.posterior.kappa / (2 * np.pi)) - 1.)) -
    #                         self.posterior.invwishart.entropy())
    #
    #     # see Eq. 10.74 in Bishop, we aren't summing over K
    #     p_avgengy = 0.5 * (self.dim * np.log(self.prior.kappa / (2. * np.pi)) + loglmbdatilde
    #                        - self.dim * self.prior.kappa / self.posterior.kappa - self.prior.kappa * self.posterior.invwishart.nu
    #                        * np.dot(self.posterior.gaussian.mu - self.prior.gaussian.mu,
    #                                 np.linalg.solve(self.posterior.invwishart.psi,
    #                                                 self.posterior.gaussian.mu - self.prior.gaussian.mu))) \
    #                 - self.prior.invwishart.log_partition() \
    #                 + (
    #                             self.prior.invwishart.nu - self.dim - 1.) / 2. * loglmbdatilde - 0.5 * self.posterior.invwishart.nu \
    #                 * np.linalg.solve(self.posterior.invwishart.psi, self.prior.invwishart.psi).trace()
    #
    #     return p_avgengy + q_entropy

    def expected_log_likelihood(self, xy=None, stats=None):
        assert isinstance(xy, (tuple, np.ndarray)) ^ isinstance(stats, tuple)

        dout = self.dout
        trash1, trash2, trash3, trash4, E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = self.posterior.get_expected_statistics()

        if self.affine:
            E_Sigmainv_A, E_Sigmainv_b = E_Sigmainv_A[:, :-1], E_Sigmainv_A[:, -1]
            E_AT_Sigmainv_A, E_AT_Sigmainv_b, E_bT_Sigmainv_b = E_AT_Sigmainv_A[:-1, :-1], E_AT_Sigmainv_A[:-1, -1], E_AT_Sigmainv_A[-1, -1]

        x, y = (xy[:, :-dout], xy[:, -dout:]) if isinstance(xy, np.ndarray) else xy

        parammat = -1. / 2 * blockarray([[E_AT_Sigmainv_A, -E_Sigmainv_A.T], [-E_Sigmainv_A, E_Sigmainv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum('ni,ni->n', xy.dot(parammat), xy)
        else:
            out = np.einsum(contract, x.dot(parammat[:-dout, :-dout]), x)
            out += np.einsum(contract, y.dot(parammat[-dout:, -dout:]), y)
            out += 2. * np.einsum(contract, x.dot(parammat[:-dout, -dout:]), y)

        out += - dout / 2. * np.log(2 * np.pi) + 1. / 2 * E_logdetSigmainv

        if self.affine:
            out += y.dot(E_Sigmainv_b)
            out -= x.dot(E_AT_Sigmainv_b)
            out -= 1. / 2 * E_bT_Sigmainv_b

        x = np.reshape(x, (-1, self.dim)) - self.posterior.gaussian.mu
        xs = np.linalg.solve(self.posterior.invwishart_niw.psi_chol, x.T)

        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        # sneaky gaussian/quadratic identity hidden here
        out2 = 0.5 * self._loglmbdatilde() - self.dim / (2. * self.posterior.kappa) - \
               self.posterior.invwishart_niw.nu / 2. * inner1d(xs.T, xs.T) - self.dim / 2. * np.log(2. * np.pi)

        return out + out2
    #
    # def expected_log_likelihood(self, x):
    #     x = np.reshape(x, (-1, self.dim)) - self.posterior.gaussian.mu
    #     xs = np.linalg.solve(self.posterior.invwishart.psi_chol, x.T)
    #
    #     # see Eqs. 10.64, 10.67, and 10.71 in Bishop
    #     # sneaky gaussian/quadratic identity hidden here
    #     return 0.5 * self._loglmbdatilde() - self.dim / (2. * self.posterior.kappa) - \
    #            self.posterior.invwishart.nu / 2. * inner1d(xs.T, xs.T) - self.dim / 2. * np.log(2. * np.pi)