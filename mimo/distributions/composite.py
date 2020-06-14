import numpy as np

from scipy.special import multigammaln, digamma
from scipy.special._ufuncs import gammaln

from mimo.abstraction import Distribution
from mimo.abstraction import Statistics as Stats

from mimo.distributions import Gaussian
from mimo.distributions import DiagonalGaussian
from mimo.distributions import InverseWishart
from mimo.distributions import InverseGamma
from mimo.distributions import MatrixNormal

from mimo.util.matrix import inv_psd


class NormalInverseWishart(Distribution):

    def __init__(self, mu, kappa, psi, nu):
        self.gaussian = Gaussian(mu=mu)
        self.invwishart = InverseWishart(psi=psi, nu=nu)
        self.kappa = kappa

    @property
    def dim(self):
        return self.gaussian.dim

    @property
    def params(self):
        return self.gaussian.mu, self.kappa, self.invwishart.psi, self.invwishart.nu

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappa, self.invwishart.psi, self.invwishart.nu = values

    def rvs(self, size=1):
        sigma = self.invwishart.rvs()
        self.gaussian.sigma = sigma / self.kappa
        mu = self.gaussian.rvs()
        return mu, sigma

    def mean(self):
        return self.gaussian.mean(), self.invwishart.mean()

    def mode(self):
        return self.gaussian.mode(), self.invwishart.mode()

    def log_likelihood(self, x):
        mu, sigma = x
        return Gaussian(mu=self.gaussian.mu,
                        sigma=sigma / self.kappa).log_likelihood(mu)\
               + self.invwishart.log_likelihood(sigma)

    def log_partition(self, params=None):
        mu, kappa, psi, nu = params if params is not None else self.params
        # return 0.5 * nu * self.dim * np.log(2)\
        #        + multigammaln(nu / 2., self.dim)\
        #        + 0.5 * self.dim * np.log(2. * np.pi / kappa)\
        #        - 0.5 * nu * np.linalg.slogdet(near_pd(psi))[1]
        return 0.5 * nu * self.dim * np.log(2)\
               + multigammaln(nu / 2., self.dim)\
               + 0.5 * self.dim * np.log(2. * np.pi / kappa)\
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
        mu = params[1] * params[0]
        kappa = params[1]
        psi = params[2] + params[1] * np.outer(params[0], params[0])
        nu = params[3]
        return Stats([mu, kappa, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        mu = natparam[0] / natparam[1]
        kappa = natparam[1]
        psi = natparam[2] - kappa * np.outer(mu, mu)
        nu = natparam[3]
        return mu, kappa, psi, nu

    def get_expected_statistics(self):
        E_J = self.invwishart.nu * np.linalg.inv(self.invwishart.psi)
        E_h = self.invwishart.nu * np.linalg.solve(self.invwishart.psi, self.gaussian.mu)
        E_muJmuT = self.dim / self.kappa + self.gaussian.mu.dot(E_h)
        E_logdetSigmainv = np.sum(digamma((self.invwishart.nu - np.arange(self.dim)) / 2.))\
                           + self.dim * np.log(2.) - np.linalg.slogdet(self.invwishart.psi)[1]

        return E_J, E_h, E_muJmuT, E_logdetSigmainv


class NormalInverseGamma(Distribution):

    def __init__(self, mu, kappas, alphas, betas):
        self.gaussian = DiagonalGaussian(mu=mu)
        self.invgamma = InverseGamma(alphas=alphas, betas=betas)
        self.kappas = kappas

    @property
    def dim(self):
        return self.gaussian.dim

    @property
    def params(self):
        return self.gaussian.mu, self.kappas,\
               self.invgamma.alphas, self.invgamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas,\
        self.invgamma.alphas, self.invgamma.betas = values

    def rvs(self, size=1):
        sigmas = self.invgamma.rvs()
        self.gaussian.sigma = np.diag(sigmas / self.kappas)
        mu = self.gaussian.rvs()
        return mu, sigmas

    def mean(self):
        return self.gaussian.mean(), self.invgamma.mean()

    def mode(self):
        return self.gaussian.mode(), self.invgamma.mode()

    def log_likelihood(self, x):
        mu, sigmas = x
        return DiagonalGaussian(mu=self.gaussian.mu,
                                sigmas=sigmas / self.kappa).log_likelihood(mu) +\
               self.invgamma.log_likelihood(sigmas)

    def log_partition(self, params=None):
        mu, kappas, alphas, betas = params if params is not None else self.params
        return np.sum(gammaln(alphas) - alphas * np.log(betas))\
               + np.sum(0.5 * np.log(2. * np.pi / kappas))

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
        mu, kappas, alphas, betas = params
        return Stats([kappas * mu, kappas, 2. * alphas,
                      2. * betas + kappas * mu**2])

    @staticmethod
    def nat_to_std(natparam):
        kappas = natparam[1]
        mu = natparam[0] / kappas
        alphas = natparam[2] / 2.
        betas = (natparam[3] - kappas * mu**2) / 2.
        return mu, kappas, alphas, betas

    def get_expected_statistics(self):
        return np.array([self.gaussian.mu * self.invgamma.alphas / self.invgamma.betas,
                         - 0.5 * (1. / self.kappas + self.gaussian.mu ** 2 * self.invgamma.alphas / self.invgamma.betas),
                         - 0.5 * (np.log(self.invgamma.betas) - digamma(self.invgamma.alphas)),
                         - 0.5 * self.invgamma.alphas / self.invgamma.betas])


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
        self.invwishart = InverseWishart(psi=psi, nu=nu)
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
        Vinv = inv_psd(params[1])
        psi = params[2] + params[0].dot(Vinv).dot(params[0].T)
        M = params[0].dot(Vinv)
        V = Vinv
        nu = params[3]
        return Stats([M, V, psi, nu])

    @staticmethod
    def nat_to_std(natparam):
        # (yxT, xxT, yyT, n)
        nu = natparam[3]
        V = inv_psd(natparam[1])
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