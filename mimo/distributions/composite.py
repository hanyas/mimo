import numpy as np

from scipy.special import multigammaln, digamma

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import Wishart
from mimo.distributions import Gamma
from mimo.distributions import MatrixNormalWithPrecision

from mimo.util.matrix import invpd, nearpd, symmetrize
from mimo.util.data import extendlists


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
        psi = invpd(params[2]) \
              + params[1] * np.outer(params[0], params[0])
        nu = params[3] - params[2].shape[0]
        return Stats([mu, kappa, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappa = natparam[1]
        psi = invpd(natparam[2] - kappa * np.outer(mu, mu))
        nu = natparam[3] + natparam[2].shape[0]
        return mu, kappa, psi, nu

    def log_partition(self, params=None):
        _, kappa, psi, nu = params if params is not None else self.params
        dim = self.dim if params else psi.shape[0]
        return - 0.5 * dim * np.log(kappa)\
               + Wishart(psi=psi, nu=nu).log_partition()

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


class TiedNormalWisharts:

    def __init__(self, mus, kappas, psi, nu):
        self.wishart = Wishart(psi=psi, nu=nu)
        self.components = [NormalWishart(mu=_mu, kappa=_kappa, psi=psi, nu=nu)
                           for _mu, _kappa in zip(mus, kappas)]

    @property
    def params(self):
        return self.mus, self.kappas, self.psi, self.nu

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.psi, self.nu = values

    def rvs(self, size=1):
        assert size == 1
        lmbda = Wishart(psi=self.psi, nu=self.nu).rvs()
        for idx, c in enumerate(self.components):
            c.gaussian.lmbda = c.kappa * lmbda
        mus = [c.gaussian.rvs() for c in self.components]
        return mus, lmbda

    @property
    def dim(self):
        return self.psi.shape[0]

    @property
    def size(self):
        return len(self.components)

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
        return self.wishart.psi

    @psi.setter
    def psi(self, value):
        self.wishart.psi = value
        for c in self.components:
            c.wishart.psi = value

    @property
    def nu(self):
        return self.wishart.nu

    @nu.setter
    def nu(self, value):
        self.wishart.nu = value
        for c in self.components:
            c.wishart.nu = value

    def mean(self):
        mus = [c.gaussian.mean() for c in self.components]
        lmbda = Wishart(psi=self.psi, nu=self.nu).mean()
        return mus, lmbda

    def mode(self):
        mus = [c.gaussian.mode() for c in self.components]
        lmbda = self.wishart.mode()
        return mus, lmbda

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        nat = [NormalWishart.std_to_nat(_params)
               for _params in zip(*extendlists(params))]
        return Stats(nat)

    @staticmethod
    def nat_to_std(natparam):
        mus, kappas = [], []
        psis, nus = [], []
        for _natparam in natparam:
            mus.append(_natparam[0] / _natparam[1])
            kappas.append(_natparam[1])
            psis.append(_natparam[2] - kappas[-1] * np.outer(mus[-1], mus[-1]))
            nus.append(_natparam[3] + _natparam[2].shape[0])

        psi = invpd(np.mean(np.stack(psis, axis=2), axis=2))
        nu = np.mean(np.hstack(nus))

        return mus, kappas, psi, nu

    def entropy(self):
        return np.sum([c.entropy() for c in self.components])

    def cross_entropy(self, dist):
        return np.sum([c.cross_entropy(d) for c, d in zip(self.components, dist.components)])

    def expected_log_likelihood(self, x):
        return np.stack([c.expected_log_likelihood(x) for c in self.components], axis=1)


class MatrixNormalWishart(Distribution):

    def __init__(self, M, K, psi, nu):
        self.matnorm = MatrixNormalWithPrecision(M=M, K=K)
        self.wishart = Wishart(psi=psi, nu=nu)

    @property
    def dcol(self):
        return self.matnorm.dcol

    @property
    def drow(self):
        return self.matnorm.drow

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu = values

    def rvs(self, size=1):
        lmbda = self.wishart.rvs()
        self.matnorm.V = lmbda
        A = self.matnorm.rvs()
        return A, lmbda

    def mean(self):
        return self.matnorm.mean(), self.wishart.mean()

    def mode(self):
        return self.matnorm.mode(), self.wishart.mode()

    def log_likelihood(self, x):
        A, lmbda = x
        return MatrixNormalWithPrecision(M=self.matnorm.M, V=lmbda,
                                         K=self.matnorm.K).log_likelihood(A)\
               + self.wishart.log_likelihood(lmbda)

    @property
    def base(self):
        return self.matnorm.base * self.wishart.base

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
        # (yxT, xxT, yyT, n)
        M = params[0].dot(params[1])
        K = params[1]
        psi = invpd(params[2]) + params[0].dot(K).dot(params[0].T)
        nu = params[3] - params[2].shape[0]
        return Stats([M, K, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        # (yxT, xxT, yyT, n)
        M = np.linalg.solve(natparam[1], natparam[0].T).T
        K = natparam[1]
        psi = invpd(natparam[2] - M.dot(K).dot(M.T))
        nu = natparam[3] + natparam[2].shape[0]

        return M, K, psi, nu

    def log_partition(self, params=None):
        M, K, psi, nu = params if params is not None else self.params
        drow = self.drow if params else M.shape[0]
        return - 0.5 * drow * np.linalg.slogdet(K)[1]\
               + Wishart(psi=psi, nu=nu).log_partition()

    def expected_statistics(self):
        E_Lmbda_A = self.wishart.nu * self.wishart.psi @ self.matnorm.M
        E_AT_Lmbda_A = - 0.5 * (self.drow * self.matnorm.K + self.matnorm.M.T.dot(E_Lmbda_A))
        E_lmbda = - 0.5 * (self.wishart.nu * self.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.wishart.nu - np.arange(self.drow)) / 2.))
                                + self.drow * np.log(2.) + 2. * np.sum(np.log(np.diag(self.wishart.psi_chol))))

        return E_Lmbda_A, E_AT_Lmbda_A, E_lmbda, E_logdet_lmbda

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.tensordot(nat_param[0], stats[0])
                  + np.tensordot(nat_param[1], stats[1])
                  + np.tensordot(nat_param[2], stats[2])
                  + nat_param[3] * stats[3])

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base()\
               - (np.tensordot(nat_param[0], stats[0])
                  + np.tensordot(nat_param[1], stats[1])
                  + np.tensordot(nat_param[2], stats[2])
                  + nat_param[3] * stats[3])

    def expected_log_likelihood(self, y, x, affine=True):
        # Natural parameter of marginal log-distirbution
        # are the expected statsitics of the posterior
        nat_param = self.expected_statistics()

        # Data statistics under a Gaussian likelihood
        # log-parition is subsumed into nat*stats
        _A = np.empty((y.shape[-1], x.shape[-1]))
        liklihood = LinearGaussianWithPrecision(A=_A, affine=affine)
        stats = liklihood.statistics(y, x, keepdim=True)
        log_base = liklihood.log_base()

        return log_base + np.einsum('kh,nkh->n', nat_param[0], stats[0])\
               + np.einsum('kh,nkh->n', nat_param[1], stats[1])\
               + np.einsum('kh,nkh->n', nat_param[2], stats[2])\
               + nat_param[3] * stats[3]


if __name__ == "__main__":
    dist = MatrixNormalWishart(M=np.zeros((3, 5)),
                               K=1e-2 * np.zeros((5, 5)),
                               psi=1. * np.eye(3, 3),
                               nu=4)

    x = np.random.randn(100, 5)
    y = np.random.randn(100, 3)

    print(dist.expected_log_likelihood(y, x, affine=False))