from abc import ABC
from functools import partial

import numpy as np
from scipy.special import digamma

from mimo.utils.abstraction import Statistics as Stats

from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithDiagonalPrecision

from mimo.distributions import Wishart
from mimo.distributions import Gamma

from mimo.distributions import MatrixNormalWithPrecision
from mimo.distributions import MatrixNormalWithDiagonalPrecision


class NormalWishart:

    def __init__(self, dim, mu=None, kappa=None,
                 psi=None, nu=None):

        self.dim = dim

        self.gaussian = GaussianWithPrecision(dim=dim, mu=mu)
        self.wishart = Wishart(dim=dim, psi=psi, nu=nu)
        self.kappa = kappa

    @property
    def params(self):
        return self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappa, self.wishart.psi, self.wishart.nu = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        # stats = [mu.T @ lmbda,
        #          -0.5 * lmbda @ (mu @ mu.T),
        #          -0.5 * lmbda,
        #          0.5 * logdet(lmbda)]
        #
        # nats = [kappa * m,
        #         kappa,
        #         psi^-1 + kappa * (m @ m.T),
        #         nu - d]

        a = params[1] * params[0]
        b = params[1]
        c = np.linalg.inv(params[2]) + params[1] * np.outer(params[0], params[0])
        d = params[3] - self.dim
        return Stats([a, b, c, d])

    def nat_to_std(self, natparam):
        mu = natparam[0] / natparam[1]
        kappa = natparam[1]
        psi = np.linalg.inv(natparam[2] - kappa * np.outer(mu, mu))
        nu = natparam[3] + self.dim
        return mu, kappa, psi, nu

    def mean(self):
        return self.gaussian.mean(), self.wishart.mean()

    def mode(self):
        mu = self.gaussian.mode()
        lmbda = (self.wishart.nu - self.dim) * self.wishart.psi
        return mu, lmbda

    def rvs(self):
        lmbda = self.wishart.rvs()
        self.gaussian.lmbda = self.kappa * lmbda
        mu = self.gaussian.rvs()
        return mu, lmbda

    @property
    def base(self):
        return self.gaussian.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, kappa, psi, nu = self.params
        return - 0.5 * self.dim * np.log(kappa)\
               + Wishart(dim=self.dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        mu, lmbda = x
        return GaussianWithPrecision(dim=self.dim, mu=self.gaussian.mu,
                                     lmbda=self.kappa * lmbda).log_likelihood(mu) \
               + self.wishart.log_likelihood(lmbda)

    def expected_statistics(self):
        # stats = [mu.T @ lmbda,
        #          -0.5 * lmbda @ (mu @ mu.T),
        #          -0.5 * lmbda,
        #          0.5 * logdet(lmbda)]

        E_mu_lmbda = self.wishart.nu * self.wishart.psi @ self.gaussian.mu
        E_mu_lmbda_muT = - 0.5 * (self.dim / self.kappa + self.gaussian.mu.dot(E_mu_lmbda))
        E_lmbda = - 0.5 * (self.wishart.nu * self.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.wishart.nu - np.arange(self.dim)) / 2.))
                                + self.dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.wishart.psi_chol))))

        return E_mu_lmbda, E_mu_lmbda_muT, E_lmbda, E_logdet_lmbda

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.dot(nat_param[0], stats[0])
                  + nat_param[1] * stats[1]
                  + np.tensordot(nat_param[2], stats[2])
                  + nat_param[3] * stats[3])

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (np.dot(nat_param[0], stats[0])
                  + nat_param[1] * stats[1]
                  + np.tensordot(nat_param[2], stats[2])
                  + nat_param[3] * stats[3])


class StackedNormalWisharts:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 psis=None, nus=None):

        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        kappas = [None] * self.size if kappas is None else kappas
        psis = [None] * self.size if psis is None else psis
        nus = [None] * self.size if nus is None else nus
        self.dists = [NormalWishart(dim, mus[k], kappas[k],
                                    psis[k], nus[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.kappas, self.psis, self.nus

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.psis, self.nus = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.gaussian.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.gaussian.mu = value[k, ...]

    @property
    def kappas(self):
        return np.array([dist.kappa for dist in self.dists])

    @kappas.setter
    def kappas(self, value):
        for k, dist in enumerate(self.dists):
            dist.kappa = value[k, ...]

    @property
    def psis(self):
        return np.array([dist.wishart.psi for dist in self.dists])

    @psis.setter
    def psis(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.psi = value[k, ...]

    @property
    def nus(self):
        return np.array([dist.wishart.nu for dist in self.dists])

    @nus.setter
    def nus(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.nu = value[k, ...]

    def mean(self):
        means = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), means))

    def mode(self):
        modes = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), modes))

    def rvs(self):
        samples = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), samples))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def expected_statistics(self):
        stats = zip(*[dist.expected_statistics() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), stats))

    def entropy(self):
        return np.array([dist.entropy() for dist in self.dists])

    def cross_entropy(self, other):
        return np.array([p.cross_entropy(q) for p, q in zip(self.dists, other.dists)])


class TiedNormalWisharts(StackedNormalWisharts, ABC):

    def __init_(self, size, dim,
                 mus=None, kappas=None,
                 psis=None, nus=None):

        super(TiedNormalWisharts, self).__init__(size, dim,
                                                 mus, kappas,
                                                 psis, nus)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        mus = np.einsum('k,kd->kd', 1. / natparam[1], natparam[0])
        kappas = natparam[1]
        psi = np.linalg.inv(np.mean(natparam[2] - np.einsum('k,kd,kl->kdl', kappas, mus, mus), axis=0))
        nu = np.mean(natparam[3] + self.dim)

        psis = np.array(self.size * [psi])
        nus = np.array(self.size * [nu])
        return mus, kappas, psis, nus


class NormalGamma:

    def __init__(self, dim, mu=None, kappas=None,
                 alphas=None, betas=None):

        self.dim = dim

        self.gaussian = GaussianWithDiagonalPrecision(dim=dim, mu=mu)
        self.gamma = Gamma(dim=dim, alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def params(self):
        return self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas, self.gamma.alphas, self.gamma.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [mu * lmbda_diag,
        #          -0.5 * lmbda_diag * mu * mu,
        #          0.5 * log(lmbda_diag),
        #          -0.5 * lmbda_diag]
        #
        # nats = [kappa * m,
        #         kappa,
        #         2 * alpha - 1,
        #         2 * beta + kappa * m * m]

        a = params[1] * params[0]
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + params[1] * params[0]**2
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappas = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - kappas * mu**2)
        return mu, kappas, alphas, betas

    def mean(self):
        return self.gaussian.mean(), self.gamma.mean()

    def mode(self):
        mu = self.gaussian.mode()
        lmbda_diag = (self.gamma.alphas - 1. / 2.) / self.gamma.betas
        return mu, lmbda_diag

    def rvs(self):
        lmbda_diag = self.gamma.rvs()
        self.gaussian.lmbda_diag = self.kappas * lmbda_diag
        mu = self.gaussian.rvs()
        return mu, lmbda_diag

    @property
    def base(self):
        return self.gaussian.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, kappas, alphas, betas = self.params
        return - 0.5 * np.sum(np.log(kappas))\
               + Gamma(dim=self.dim, alphas=alphas, betas=betas).log_partition()

    def log_likelihood(self, x):
        mu, lmbda_diag = x
        return GaussianWithDiagonalPrecision(dim=self.dim, mu=self.gaussian.mu,
                                             lmbda_diag=self.kappas * lmbda_diag).log_likelihood(mu)\
               + self.gamma.log_likelihood(lmbda_diag)

    def expected_statistics(self):
        # stats = [mu * lmbda_diag,
        #          -0.5 * lmbda_diag * mu * mu,
        #          0.5 * log(lmbda_diag),
        #          -0.5 * lmbda_diag]

        E_mu_lmbdas = self.gamma.alphas / self.gamma.betas * self.gaussian.mu
        E_lmbdas_mu_mu = - 0.5 * (1. / self.kappas + self.gaussian.mu * E_mu_lmbdas)
        E_log_lmbdas = 0.5 * (digamma(self.gamma.alphas) - np.log(self.gamma.betas))
        E_lmbdas = - 0.5 * (self.gamma.alphas / self.gamma.betas)

        return E_mu_lmbdas, E_lmbdas_mu_mu, E_log_lmbdas, E_lmbdas

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.dot(nat_param[0], stats[0])
                  + np.dot(nat_param[1], stats[1])
                  + np.dot(nat_param[2], stats[2])
                  + np.dot(nat_param[3], stats[3]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (np.dot(nat_param[0], stats[0])
                  + np.dot(nat_param[1], stats[1])
                  + np.dot(nat_param[2], stats[2])
                  + np.dot(nat_param[3], stats[3]))


class StackedNormalGammas:

    def __init__(self, size, dim,
                 mus=None, kappas=None,
                 alphas=None, betas=None):

        self.size = size
        self.dim = dim

        mus = [None] * self.size if mus is None else mus
        kappas = [None] * self.size if kappas is None else kappas
        alphas = [None] * self.size if alphas is None else alphas
        betas = [None] * self.size if betas is None else betas
        self.dists = [NormalGamma(dim, mus[k], kappas[k],
                                  alphas[k], betas[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.mus, self.kappas, self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.mus, self.kappas, self.alphas, self.betas = values

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def mus(self):
        return np.array([dist.gaussian.mu for dist in self.dists])

    @mus.setter
    def mus(self, value):
        for k, dist in enumerate(self.dists):
            dist.gaussian.mu = value[k, ...]

    @property
    def kappas(self):
        return np.array([dist.kappas for dist in self.dists])

    @kappas.setter
    def kappas(self, value):
        for k, dist in enumerate(self.dists):
            dist.kappas = value[k, ...]

    @property
    def alphas(self):
        return np.array([dist.gamma.alphas for dist in self.dists])

    @alphas.setter
    def alphas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.psi = value[k, ...]

    @property
    def betas(self):
        return np.array([dist.gamma.betas for dist in self.dists])

    @betas.setter
    def betas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.nu = value[k, ...]

    def mean(self):
        means = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), means))

    def mode(self):
        modes = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), modes))

    def rvs(self):
        samples = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), samples))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def expected_statistics(self):
        stats = zip(*[dist.expected_statistics() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), stats))

    def entropy(self):
        return np.array([dist.entropy() for dist in self.dists])

    def cross_entropy(self, other):
        return np.array([p.cross_entropy(q) for p, q in zip(self.dists, other.dists)])


class TiedNormalGammas(StackedNormalGammas, ABC):

    def __init_(self, size, dim,
                 mus=None, kappas=None,
                 alphas=None, betas=None):

        super(TiedNormalGammas, self).__init__(size, dim,
                                               mus, kappas,
                                               alphas, betas)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        mus = np.einsum('kd,kd->kd', 1. / natparam[1], natparam[0])
        kappas = natparam[1]
        alphas = np.mean(0.5 * (natparam[2] + 1), axis=0)
        betas = np.mean(0.5 * (natparam[3] - kappas * mus**2), axis=0)

        alphas = np.array(self.size * [alphas])
        betas = np.array(self.size * [betas])
        return mus, kappas, alphas, betas


class MatrixNormalWishart:

    def __init__(self, column_dim, row_dim,
                 M=None, K=None, psi=None, nu=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.matnorm = MatrixNormalWithPrecision(column_dim, row_dim, M=M, K=K)
        self.wishart = Wishart(dim=row_dim, psi=psi, nu=nu)

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, self.wishart.psi, self.wishart.nu = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        # stats = [A.T @ V,
        #          -0.5 * A.T @ V @ A,
        #          -0.5 * V,
        #          0.5 * log_det(V)]
        #
        # nats = [M @ K,
        #         K,
        #         psi^-1 + M @ K @ M.T,
        #         nu - d - 1. + l]

        a = params[0] @ params[1]
        b = params[1]
        c = np.linalg.inv(params[2]) + params[0] @ params[1] @ params[0].T
        d = params[3] - self.row_dim - 1. + self.column_dim
        return Stats([a, b, c, d])

    def nat_to_std(self, natparam):
        M = natparam[0] @ np.linalg.inv(natparam[1])
        K = natparam[1]
        psi = np.linalg.inv(natparam[2] - M @ K @ M.T)
        nu = natparam[3] + self.row_dim + 1. - self.column_dim
        return M, K, psi, nu

    def mean(self):
        return self.matnorm.mean(), self.wishart.mean()

    def mode(self):
        A = self.matnorm.mode()
        lmbda = (self.wishart.nu - self.row_dim) * self.wishart.psi
        return A, lmbda

    def rvs(self, size=1):
        lmbda = self.wishart.rvs()
        self.matnorm.V = lmbda
        A = self.matnorm.rvs()
        return A, lmbda

    @property
    def base(self):
        return self.matnorm.base * self.wishart.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, K, psi, nu = self.params
        return - 0.5 * self.row_dim * np.linalg.slogdet(K)[1]\
               + Wishart(dim=self.row_dim, psi=psi, nu=nu).log_partition()

    def log_likelihood(self, x):
        A, lmbda = x
        return MatrixNormalWithPrecision(column_dim=self.column_dim,
                                         row_dim=self.row_dim,
                                         M=self.matnorm.M, V=lmbda,
                                         K=self.matnorm.K).log_likelihood(A)\
               + self.wishart.log_likelihood(lmbda)

    def expected_statistics(self):
        # stats = [A.T @ V,
        #          -0.5 * A.T @ V @ A,
        #          -0.5 * V,
        #          0.5 * log_det(V)]

        E_Lmbda_A = self.wishart.nu * self.wishart.psi @ self.matnorm.M
        E_AT_Lmbda_A = - 0.5 * (self.row_dim * np.linalg.inv(self.matnorm.K) + self.matnorm.M.T.dot(E_Lmbda_A))
        E_lmbda = - 0.5 * (self.wishart.nu * self.wishart.psi)
        E_logdet_lmbda = 0.5 * (np.sum(digamma((self.wishart.nu - np.arange(self.row_dim)) / 2.))
                                + self.row_dim * np.log(2.) + 2. * np.sum(np.log(np.diag(self.wishart.psi_chol))))

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
        return dist.log_partition() - dist.log_base() \
               - (np.tensordot(nat_param[0], stats[0])
                  + np.tensordot(nat_param[1], stats[1])
                  + np.tensordot(nat_param[2], stats[2])
                  + nat_param[3] * stats[3])


class StackedMatrixNormalWisharts:

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, psis=None, nus=None):

        self.size = size

        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        psis = [None] * self.size if psis is None else psis
        nus = [None] * self.size if nus is None else nus

        self.dists = [MatrixNormalWishart(column_dim, row_dim,
                                          Ms[k], Ks[k], psis[k], nus[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Ks, self.psis, self.nus

    @params.setter
    def params(self, values):
        self.Ms, self.Ks, self.psis, self.nus = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def Ms(self):
        return np.array([dist.matnorm.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.M = value[k, ...]

    @property
    def Ks(self):
        return np.array([dist.matnorm.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.K = value[k, ...]

    @property
    def psis(self):
        return np.array([dist.wishart.psi for dist in self.dists])

    @psis.setter
    def psis(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.psi = value[k, ...]

    @property
    def nus(self):
        return np.array([dist.wishart.nu for dist in self.dists])

    @nus.setter
    def nus(self, value):
        for k, dist in enumerate(self.dists):
            dist.wishart.nu = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def expected_statistics(self):
        stats = zip(*[dist.expected_statistics() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), stats))

    def entropy(self):
        return np.array([dist.entropy() for dist in self.dists])

    def cross_entropy(self, other):
        return np.array([p.cross_entropy(q) for p, q in zip(self.dists, other.dists)])


class TiedMatrixNormalWisharts(StackedMatrixNormalWisharts):

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, psis=None, nus=None):

        super(TiedMatrixNormalWisharts, self).__init__(size, column_dim, row_dim,
                                                       Ms, Ks, psis, nus)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        Ms = np.einsum('kdl,klh->kdh', natparam[0], np.linalg.inv(natparam[1]))
        Ks = natparam[1]
        psi = np.linalg.inv(np.mean(natparam[2] - np.einsum('kdl,klm,khm->kdh', Ms, Ks, Ms), axis=0))
        nu = np.mean(natparam[3] + self.row_dim + 1 - self.column_dim)

        psis = np.array(self.size * [psi])
        nus = np.array(self.size * [nu])
        return Ms, Ks, psis, nus


class MatrixNormalGamma:

    def __init__(self, column_dim, row_dim,
                 M=None, K=None, alphas=None, betas=None):

        self.column_dim = column_dim
        self.row_dim = row_dim

        self.matnorm = MatrixNormalWithDiagonalPrecision(column_dim, row_dim, M=M, K=K)
        self.gamma = Gamma(dim=row_dim, alphas=alphas, betas=betas)

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.K, self.gamma.alphas, self.gamma.betas

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.K, self.gamma.alphas, self.gamma.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    @staticmethod
    def std_to_nat(params):
        # stats = [A.T * V_diag,
        #          -0.5 * A.T @ A,
        #          0.5 * log(V_diag),
        #          -0.5 * V_diag]
        #
        # nats = [M @ K,
        #         K,
        #         2. * alpha - 1.,
        #         2. * beta + M @ K @ M.T]

        a = params[0] @ params[1]
        b = params[1]
        c = 2. * params[2] - 1.
        d = 2. * params[3] + np.einsum('dl,lm,dm->d', params[0], params[1], params[0])
        return Stats([a, b, c, d])

    @staticmethod
    def nat_to_std(natparam):
        M = natparam[0] @ np.linalg.inv(natparam[1])
        K = natparam[1]
        alphas = 0.5 * (natparam[2] + 1.)
        betas = 0.5 * (natparam[3] - np.einsum('dl,lm,dm->d', M, K, M))
        return M, K, alphas, betas

    def mean(self):
        return self.matnorm.mean(), self.gamma.mean()

    def mode(self):
        A = self.matnorm.mode()
        lmbda_diag = (self.gamma.alphas - 1. / 2.) / self.gamma.betas
        return A, lmbda_diag

    def rvs(self, size=1):
        lmbdas = self.gamma.rvs()
        self.matnorm.V_diag = lmbdas
        A = self.matnorm.rvs()
        return A, lmbdas

    @property
    def base(self):
        return self.matnorm.base * self.gamma.base

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        _, K, alphas, betas = self.params
        return - self.row_dim * (0.5 * self.column_dim * np.linalg.slogdet(K)[1])\
               + Gamma(dim=self.row_dim, alphas=alphas, betas=betas).log_partition()

    def log_likelihood(self, x):
        A, lmbda_diag = x
        return MatrixNormalWithDiagonalPrecision(column_dim=self.column_dim,
                                                 row_dim=self.row_dim,
                                                 M=self.matnorm.M, V_diag=lmbda_diag,
                                                 K=self.matnorm.K).log_likelihood(A)\
               + self.gamma.log_likelihood(lmbda_diag)

    def expected_statistics(self):
        # stats = [A.T * V_diag,
        #          -0.5 * A.T @ A,
        #          0.5 * log(V_diag),
        #          -0.5 * V_diag]

        E_lmbdas_A = np.diag(self.gamma.alphas / self.gamma.betas) @ self.matnorm.M
        E_AT_lmbdas_A = - 0.5 * (self.row_dim * np.linalg.inv(self.matnorm.K) + self.matnorm.M.T.dot(E_lmbdas_A))
        E_log_lmbdas = 0.5 * (digamma(self.gamma.alphas) - np.log(self.gamma.betas))
        E_lmbdas = - 0.5 * (self.gamma.alphas / self.gamma.betas)

        return E_lmbdas_A, E_AT_lmbdas_A, E_log_lmbdas, E_lmbdas

    def entropy(self):
        nat_param, stats = self.nat_param, self.expected_statistics()
        return self.log_partition() - self.log_base()\
               - (np.tensordot(nat_param[0], stats[0])
                  + np.tensordot(nat_param[1], stats[1])
                  + nat_param[2] * stats[2]
                  + np.tensordot(nat_param[3], stats[3]))

    def cross_entropy(self, dist):
        nat_param, stats = dist.nat_param, self.expected_statistics()
        return dist.log_partition() - dist.log_base() \
               - (np.tensordot(nat_param[0], stats[0])
                  + np.tensordot(nat_param[1], stats[1])
                  + nat_param[2] * stats[2]
                  + np.tensordot(nat_param[3], stats[3]))


class StackedMatrixNormalGammas:

    def __init__(self, size,
                 column_dim, row_dim,
                 Ms=None, Ks=None,
                 alphas=None, betas=None):

        self.size = size

        self.column_dim = column_dim
        self.row_dim = row_dim

        Ms = [None] * self.size if Ms is None else Ms
        Ks = [None] * self.size if Ks is None else Ks
        alphas = [None] * self.size if alphas is None else alphas
        betas = [None] * self.size if betas is None else betas

        self.dists = [MatrixNormalGamma(column_dim, row_dim,
                                        Ms[k], Ks[k], alphas[k], betas[k])
                      for k in range(self.size)]

    @property
    def params(self):
        return self.Ms, self.Ks, self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.Ms, self.Ks, self.alphas, self.betas = values

    @property
    def nat_param(self):
        return self.std_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_std(natparam)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        natparams_list = list(zip(*natparam))
        params_list = [dist.nat_to_std(par) for dist, par in zip(self.dists, natparams_list)]
        params_stack = tuple(map(partial(np.stack, axis=0), zip(*params_list)))
        return params_stack

    @property
    def Ms(self):
        return np.array([dist.matnorm.M for dist in self.dists])

    @Ms.setter
    def Ms(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.M = value[k, ...]

    @property
    def Ks(self):
        return np.array([dist.matnorm.K for dist in self.dists])

    @Ks.setter
    def Ks(self, value):
        for k, dist in enumerate(self.dists):
            dist.matnorm.K = value[k, ...]

    @property
    def alphas(self):
        return np.array([dist.gamma.alphas for dist in self.dists])

    @alphas.setter
    def alphas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.alphas = value[k, ...]

    @property
    def betas(self):
        return np.array([dist.gamma.betas for dist in self.dists])

    @betas.setter
    def betas(self, value):
        for k, dist in enumerate(self.dists):
            dist.gamma.betas = value[k, ...]

    def mean(self):
        zipped = zip(*[dist.mean() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def mode(self):
        zipped = zip(*[dist.mode() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    def rvs(self):
        zipped = zip(*[dist.rvs() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), zipped))

    @property
    def base(self):
        return np.array([dist.base for dist in self.dists])

    def log_base(self):
        return np.log(self.base)

    def log_partition(self):
        return np.array([dist.log_partition() for dist in self.dists])

    def log_likelihood(self, x):
        return np.sum([dist.log_likelihood(_x)
                       for dist, _x in zip(self.dists, list(zip(*x)))])

    def expected_statistics(self):
        stats = zip(*[dist.expected_statistics() for dist in self.dists])
        return tuple(map(partial(np.stack, axis=0), stats))

    def entropy(self):
        return np.array([dist.entropy() for dist in self.dists])

    def cross_entropy(self, other):
        return np.array([p.cross_entropy(q) for p, q in zip(self.dists, other.dists)])


class TiedMatrixNormalGammas(StackedMatrixNormalGammas):

    def __init__(self, size, column_dim, row_dim,
                 Ms=None, Ks=None, alphas=None, betas=None):

        super(TiedMatrixNormalGammas, self).__init__(size, column_dim, row_dim,
                                                     Ms, Ks, alphas, betas)

    def std_to_nat(self, params):
        params_list = list(zip(*params))
        natparams_list = [dist.std_to_nat(par) for dist, par in zip(self.dists, params_list)]
        natparams_stack = Stats(map(partial(np.stack, axis=0), zip(*natparams_list)))
        return natparams_stack

    def nat_to_std(self, natparam):
        aT = np.transpose(natparam[0], (0, 2, 1))
        bT = np.transpose(natparam[1], (0, 2, 1))

        Ms = np.transpose(np.linalg.solve(bT, aT), (0, 2, 1))
        Ks = natparam[1]
        alphas = np.mean(0.5 * (natparam[2] + 1.), axis=0)
        betas = np.mean(0.5 * (natparam[3] - np.einsum('kdl,klm,kdm->kd', Ms, Ks, Ms)), axis=0)

        alphas = np.array(self.size * [alphas])
        betas = np.array(self.size * [betas])
        return Ms, Ks, alphas, betas
