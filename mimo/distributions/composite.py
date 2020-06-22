import numpy as np

from scipy.special import multigammaln, digamma
from scipy.special import gammaln

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import Wishart
from mimo.distributions import Gamma
from mimo.distributions import MatrixNormal

from mimo.util.matrix import inv_pd, near_pd


class NormalInverseWishart(Distribution):
    def __init__(self, mu, kappa, psi, nu):
        self.gaussian = GaussianWithPrecision(mu=mu)
        self.wishart = Wishart(psi=psi, nu=nu)
        self.kappa = kappa


class NormalInverseGamma(Distribution):

    def __init__(self, mu, kappas, alphas, betas):
        self.gaussian = GaussianWithDiagonalPrecision(mu=mu)
        self.invgamma = Gamma(alphas=alphas, betas=betas)
        self.kappas = kappas


class NormalWishart(Distribution):

    def __init__(self, mu, kappa, psi, nu):
        self.gaussian = GaussianWithPrecision(mu=mu)
        self.wishart = Wishart(psi=psi, nu=nu)
        self.kappa = kappa

    @property
    def dim(self):
        return self.gaussian.dim

    @property
    def params(self):
        return self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu = values

    def rvs(self, size=1):
        lmbda = self.wishart.rvs()
        self.gaussian.lmbda = self.kappa * lmbda
        mu = self.gaussian.rvs()
        return mu, lmbda

    def mean(self):
        return self.gaussian.mean(), self.wishart.mean()

    def mode(self):
        return self.gaussian.mode(), self.wishart.mode()

    def log_likelihood(self, x):
        mu, lmbda = x
        return GaussianWithPrecision(mu=self.gaussian.mu,
                                     lmbda=self.kappa * np.eye(self.dim)).log_likelihood(mu) \
               + self.wishart.log_likelihood(lmbda)

    @property
    def base(self):
        return self.gaussian.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # The definition of stats is slightly different
        # from literatur to make posterior updates easy

        # Assumed stats
        # stats = [lmbda @ x,
        #          -0.5 * lmbda @ xxT,
        #          -0.5 * lmbda,
        #          0.5 * logdet_lmbda]

        mu = params[1] * params[0]
        kappa = params[1]
        psi = inv_pd(params[2])\
              + params[1] * np.outer(params[0], params[0])
        nu = params[3] - params[2].shape[0]
        return Stats([mu, kappa, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappa = natparam[1]
        psi = inv_pd(natparam[2] - kappa * np.outer(mu, mu))
        nu = natparam[3] + natparam[2].shape[0]
        return mu, kappa, psi, nu

    def log_partition(self, params=None):
        _, kappa, psi, nu = params if params is not None else self.params
        dim = self.dim if params else psi.shape[0]
        return - dim / 2. * np.log(kappa) + Wishart(psi=psi, nu=nu).log_partition()

    def expected_statistics(self):
        # stats = [lmbda @ x,
        #          -0.5 * lmbda @ xxT,
        #          -0.5 * lmbda,
        #          0.5 * logdet_lmbda]

        E_x = self.wishart.nu * self.wishart.psi @ self.gaussian.mu
        E_xLmbdaxT = - 0.5 * (self.dim / self.kappa + self.gaussian.mu.dot(E_x))
        E_lmbda = - 0.5 * (self.wishart.nu * self.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.wishart.nu - np.arange(self.dim)) / 2.))
                                + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.wishart.psi_chol))))

        return E_x, E_xLmbdaxT, E_lmbda, E_logdet_lmbda

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.dot(nat_param[0], stats[0]) + nat_param[1] * stats[1]
                  + np.tensordot(nat_param[2], stats[2]) + nat_param[3] * stats[3])

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base()\
               - (np.dot(nat_param[0], stats[0]) + nat_param[1] * stats[1]
                  + np.tensordot(nat_param[2], stats[2]) + nat_param[3] * stats[3])

    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat*stats
        liklihood = GaussianWithPrecision(mu=np.empty(x.shape[-1]))
        stats = liklihood.statistics(x, keepdim=True)
        log_base = liklihood.log_base()

        return log_base + np.einsum('k,nk->n', nat_param[0], stats[0])\
               + nat_param[1] * stats[1] + nat_param[3] * stats[3]\
               + np.einsum('kh,nkh->n', nat_param[2], stats[2])


class NormalGamma(Distribution):

    def __init__(self, mu, kappas, alphas, betas):
        self.gaussian = GaussianWithDiagonalPrecision(mu=mu)
        self.gamma = Gamma(alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def dim(self):
        return self.gaussian.dim

    @property
    def params(self):
        return self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas = values

    def rvs(self, size=1):
        lmbdas = self.gamma.rvs()
        self.gaussian.lmbdas = self.kappas * lmbdas
        mu = self.gaussian.rvs()
        return mu, lmbdas

    def mean(self):
        return self.gaussian.mean(), self.gamma.mean()

    def mode(self):
        return self.gaussian.mode(), self.gamma.mode()

    def log_likelihood(self, x):
        mu, lmbdas = x
        return GaussianWithDiagonalPrecision(mu=self.gaussian.mu,
                                             lmbdas=self.kappas * lmbdas).log_likelihood(mu)\
               + self.gamma.log_likelihood(lmbdas)

    @property
    def base(self):
        return self.gaussian.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # The definition of stats is slightly different
        # from literatur to make posterior updates easy

        # Assumed stats
        # stats = [lmbdas * x,
        #          -0.5 * lmbdas * xx,
        #          0.5 * log_lmbdas
        #          -0.5 * lmbdas]

        mu = params[1] * params[0]
        kappas = params[1]
        alphas = 2. * params[2] - 1.
        betas = 2. * params[3] + params[1] * params[0]**2
        return Stats([mu, kappas, alphas, betas])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappas = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - kappas * mu**2)
        return mu, kappas, alphas, betas

    def log_partition(self, params=None):
        mu, kappas, alphas, betas = params if params is not None else self.params
        return - 0.5 * np.sum(np.log(kappas)) + Gamma(alphas=alphas, betas=betas).log_partition()

    def expected_statistics(self):
        # stats = [lmbdas * x,
        #          -0.5 * lmbdas * xx,
        #          0.5 * log_lmbdas
        #          -0.5 * lmbdas]

        E_x = self.gamma.alphas / self.gamma.betas * self.gaussian.mu
        E_lmbdas_xx = - 0.5 * (1. / self.kappas + self.gaussian.mu * E_x)
        E_log_lmbdas = 0.5 * (digamma(self.gamma.alphas) - np.log(self.gamma.betas))
        E_lmbdas = - 0.5 * (self.gamma.alphas / self.gamma.betas)

        return E_x, E_lmbdas_xx, E_log_lmbdas, E_lmbdas

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.dot(nat_param[0], stats[0]) + np.dot(nat_param[1], stats[1])
                  + np.dot(nat_param[2], stats[2]) + np.dot(nat_param[3], stats[3]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.dot(nat_param[0], stats[0]) + np.dot(nat_param[1], stats[1])
                  + np.dot(nat_param[2], stats[2]) + np.dot(nat_param[3], stats[3]))

    def expected_log_likelihood(self, x):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat*stats
        liklihood = GaussianWithDiagonalPrecision(mu=np.empty(x.shape[-1]))
        stats = liklihood.statistics(x, keepdim=True)
        log_base = liklihood.log_base()

        return log_base + np.einsum('k,nk->n', nat_param[0], stats[0])\
               + np.einsum('k,nk->n', nat_param[1], stats[1])\
               + np.einsum('k,nk->n', nat_param[2], stats[2])\
               + np.einsum('k,nk->n', nat_param[3], stats[3])


class TiedNormalInverseWisharts:

    def __init__(self, mus, kappas, psi, nu):
        self.components = [NormalInverseWishart(mu=_mu, kappa=_kappa, psi=psi, nu=nu)
                           for _mu, _kappa in zip(mus, kappas)]

    def rvs(self, size=1):
        assert size == 1
        sigma = self.components[0].invwishart.rvs()
        for idx, c in enumerate(self.components):
            c.gaussian.sigma = sigma / c.kappa
        mus = [c.gaussian.rvs() for c in self.components]
        return mus, sigma

    @property
    def size(self):
        return len(self.components)

    @property
    def params(self):
        return self.mus, self.kappas, self.psi, self.nu

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.psi, self.nu = values

    @property
    def mus(self):
        return[c.gaussian.mu for c in self.components]

    @mus.setter
    def mus(self, values):
        for idx, c in enumerate(self.components):
            c.gaussian.mu = values[idx]

    @property
    def kappas(self):
        return [c.kappa for c in self.components]

    @kappas.setter
    def kappas(self, values):
        for idx, c in enumerate(self.components):
            c.kappa = values[idx]

    @property
    def psi(self):
        assert np.all([c.invwishart.psi == self.components[0].invwishart.psi
                       for c in self.components])
        return self.components[0].invwishart.psi

    @psi.setter
    def psi(self, value):
        for c in self.components:
            c.invwishart.psi = value

    @property
    def nu(self):
        assert np.all([c.invwishart.nu == self.components[0].invwishart.nu
                       for c in self.components])
        return self.components[0].invwishart.nu

    @nu.setter
    def nu(self, value):
        for c in self.components:
            c.invwishart.nu = value

    def mean(self):
        mus = [c.mean()[0] for c in self.components]
        sigma = self.components[0].mean()[1]
        return mus, sigma

    def mode(self):
        mus = [c.mode()[0] for c in self.components]
        sigma = self.components[0].mode()[1]
        return mus, sigma

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        nat = []
        mus, kappas, psi, nu = params
        for mu, kappa in zip(mus, kappas):
            _mu = kappa * mu
            _kappa = kappa
            _psi = psi + kappa * np.outer(mu, mu)
            _nu = nu
            nat.append(Stats([_mu, _kappa, _psi, _nu]))
        return Stats(nat)

    @staticmethod
    def nat_to_std(natparam):
        mus, kappas, psis, nus = [], [], [], []
        for _natparam in natparam:
            mus.append(_natparam[0] / _natparam[1])
            kappas.append(_natparam[1])
            psis.append(_natparam[2] - kappas[-1] * np.outer(mus[-1], mus[-1]))
            nus.append(_natparam[3])

        psi = np.mean(np.stack(psis, axis=2), axis=-1)
        nu = np.mean(np.hstack(nus), axis=0)

        return mus, kappas, psi, nu


class MatrixNormalInverseWishart(Distribution):

    def __init__(self, M, V, affine, psi, nu):
        self.matnorm = MatrixNormal(M=M, V=V)
        self.invwishart = Wishart(psi=psi, nu=nu)
        self.affine = affine

    @property
    def dcol(self):
        return self.matnorm.dcol

    @property
    def drow(self):
        return self.matnorm.drow

    @property
    def params(self):
        return tuple([*self.matnorm.params, *self.invwishart.params])

    @params.setter
    def params(self, values):
        self.matnorm.params = values[:2]
        self.invwishart.params = values[2:]

    def rvs(self, size=1):
        sigma = self.invwishart.rvs()
        self.matnorm.U = sigma
        A = self.matnorm.rvs()
        return A, sigma

    def mean(self):
        return self.matnorm.mean(), self.invwishart.mean()

    def mode(self):
        return self.matnorm.mode(), self.invwishart.mode()

    def log_likelihood(self, x):
        A, sigma = x
        return MatrixNormal(M=self.matnorm.M, V=self.matnorm.V, U=sigma).log_likelihood(A)\
               + self.invwishart.log_likelihood(sigma)

    def log_partition(self, params=None):
        M, V, psi, nu = params if params is not None else self.params
        # return 0.5 * nu * self.drow * np.log(2)\
        #        + multigammaln(nu / 2., self.drow)\
        #        + 0.5 * self.drow * np.log(2. * np.pi)\
        #        - 0.5 * self.drow * np.linalg.slogdet(near_pd(V))[1]\
        #        - 0.5 * nu * np.linalg.slogdet(near_pd(psi))[1]
        return 0.5 * nu * self.drow * np.log(2)\
               + multigammaln(nu / 2., self.drow)\
               + 0.5 * self.drow * np.log(2. * np.pi)\
               - 0.5 * self.drow * np.linalg.slogdet(V)[1]\
               - 0.5 * nu * np.linalg.slogdet(psi)[1]

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        Vinv = inv_pd(params[1])
        psi = params[2] + params[0].dot(Vinv).dot(params[0].T)
        M = params[0].dot(Vinv)
        V = Vinv
        nu = params[3]
        return Stats([M, V, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        # (yxT, xxT, yyT, n)
        nu = natparam[3]
        V = inv_pd(natparam[1])
        M = np.linalg.solve(natparam[1], natparam[0].T).T

        # This subtraction seems unstable!
        # It does not necessarily return a PSD matrix
        psi = natparam[2] - M.dot(natparam[0].T)

        # numerical paddcolg here...
        # V = near_pd(V + 1e-16 * np.eye(V.shape[0]))
        # psi = near_pd(psi + 1e-16 * np.eye(psi.shape[0]))

        V = V + 1e-16 * np.eye(V.shape[0])
        psi = psi + 1e-16 * np.eye(psi.shape[0])

        # assert np.all(0 < np.linalg.eigvalsh(psi))
        # assert np.all(0 < np.linalg.eigvalsh(V))

        return M, V, psi, nu

    def get_expected_statistics(self):
        E_Sigmainv = self.invwishart.nu * np.linalg.inv(self.invwishart.psi)
        E_Sigmainv_A = self.invwishart.nu * np.linalg.solve(self.invwishart.psi, self.matnorm.M)
        E_AT_Sigmainv_A = self.drow * self.matnorm.V + self.invwishart.nu\
                          * self.matnorm.M.T.dot(np.linalg.solve(self.invwishart.psi, self.matnorm.M))
        E_logdetSigmainv = digamma((self.invwishart.nu - np.arange(self.drow)) / 2.).sum() \
                           + self.drow * np.log(2) - np.linalg.slogdet(self.invwishart.psi)[1]

        return E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv


if __name__ == '__main__':
    psi = np.linalg.inv(np.array([[16119.32896757,  -283.99104534],
                                  [-283.99104534,   462.54112854]]))
    nu = 103
    mu = np.array([1.59797067e+01, 1.77075092e-03])
    kappa = 100.01

    g = NormalWishart(mu=mu, kappa=kappa,
                      psi=psi, nu=nu)
    print(g.entropy())
    x = np.random.randn(5, 2)
    print(g.expected_log_likelihood(x))
