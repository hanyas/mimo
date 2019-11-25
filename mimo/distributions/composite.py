import numpy as np

from mimo.abstractions import Distribution
from mimo.distributions import InverseWishart
from mimo.distributions import InverseGamma
from mimo.distributions.gaussian import Gaussian, DiagonalGaussian
from mimo.distributions import MatrixNormal

from scipy.special import multigammaln, digamma, gammaln

from mimo.util.general import blockarray, inv_psd


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

    def rvs(self, size=None):
        # sample sigma from inverse wishart
        sigma = self.invwishart.rvs()
        # sample mean from gaussian
        self.gaussian.sigma = sigma / self.kappa
        mu = self.gaussian.rvs()
        return mu, sigma

    def log_likelihood(self, x):
        mu, sigma = x
        return Gaussian(mu=self.gaussian.mu, sigma=sigma / self.kappa).log_likelihood(mu)\
               + self.invwishart.log_likelihood(sigma)

    def mean(self):
        return tuple([self.gaussian.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invwishart.mode()])

    def log_partition(self):
        return 0.5 * self.invwishart.nu * self.dim * np.log(2)\
               + multigammaln(self.invwishart.nu / 2., self.dim)\
               + 0.5 * self.dim * np.log(2. * np.pi / self.kappa)\
               - self.invwishart.nu * np.sum(np.log(np.diag(self.invwishart.psi_chol)))

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self._standard_to_nat(self.gaussian.mu, self.kappa,
                                     self.invwishart.psi, self.invwishart.nu)

    @nat_param.setter
    def nat_param(self, natparam):
        self.gaussian.mu, self.kappa,\
        self.invwishart.psi, self.invwishart.nu = self._nat_to_standard(natparam)

    def _standard_to_nat(self, mu, kappa, psi, nu):
        _psi = psi + kappa * np.outer(mu, mu)
        _mu = kappa * mu
        _kappa = kappa
        _nu = nu + 2 + self.dim
        return np.array([_mu, _kappa, _psi, _nu])

    def _nat_to_standard(self, natparam):
        kappa = natparam[1]
        mu = natparam[0] / kappa
        nu = natparam[3] - 2 - self.dim
        psi = natparam[2] - kappa * np.outer(mu, mu)
        return mu, kappa, psi, nu

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            xxT = np.einsum('nk,nh->kh', data, data)
            x = np.sum(data, axis=0)
            n = data.shape[0]
            return np.array([x, n, xxT, n])
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            xxT = np.einsum('nk,n,nh->kh', data, weights, data)
            x = weights.dot(data)
            n = weights.sum()
            return np.array([x, n, xxT, n])
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.dim, )), 0,
                         np.zeros((self.dim, self.dim)), 0])

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
        return self.gaussian.mu, self.kappas, self.invgamma.alphas, self.invgamma.betas

    @params.setter
    def params(self, values):
        self.gaussian.mu, self.kappas, self.invgamma.alphas, self.invgamma.betas = values

    def rvs(self, size=None):
        # sample sigma from inverse wishart
        sigmas = self.invgamma.rvs()
        # sample mean from gaussian
        self.gaussian.sigma = np.diag(sigmas / self.kappas)
        mu = self.gaussian.rvs()
        return mu, sigmas

    def log_likelihood(self, x):
        mu, sigmas = x
        return DiagonalGaussian(mu=self.gaussian.mu, sigmas=sigmas/self.kappa).log_likelihood(mu)\
               + self.invgamma.log_likelihood(sigmas)

    def mean(self):
        return tuple([self.gaussian.mean(), self.invgamma.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invgamma.mode()])

    def log_partition(self):
        return np.sum(gammaln(self.invgamma.alphas)
                      - self.invgamma.alphas * np.log(self.invgamma.betas))\
               + np.sum(0.5 * np.log(2. * np.pi / self.kappas))

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self._standard_to_nat(self.gaussian.mu, self.kappas,
                                     self.invgamma.alphas, self.invgamma.betas)

    @nat_param.setter
    def nat_param(self, natparam):
        self.gaussian.mu, self.kappas,\
        self.invgamma.alphas, self.invgamma.betas = self._nat_to_standard(natparam)

    def _standard_to_nat(self, mu, kappas, alphas, betas):
        return np.array([kappas * mu, kappas, 2. * alphas, 2. * betas + kappas * mu**2])

    def _nat_to_standard(self, natparam):
        _kappas = natparam[1]
        _mu = natparam[0] / _kappas
        _alphas = natparam[2] / 2.
        _betas = (natparam[3] - _kappas * _mu**2) / 2.
        return _mu, _kappas, _alphas, _betas

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]

            xx = np.einsum('nk,nk->k', data, data)
            x = np.sum(data, axis=0)
            n = np.repeat(data.shape[0], self.dim)
            return np.array([x, n, n, xx])
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]

            xx = np.einsum('nk,n,nk->k', data, weights, data)
            x = weights.dot(data)
            n = np.repeat(weights.sum(), self.dim)
            return np.array([x, n, n, xx])
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.dim, )), np.zeros((self.dim, )),
                         np.zeros((self.dim, )), np.zeros((self.dim, ))])

    def get_expected_statistics(self):
        return np.array([self.gaussian.mu * self.invgamma.alphas / self.invgamma.betas,
                         - 0.5 * (1. / self.kappas + self.gaussian.mu ** 2 * self.invgamma.alphas / self.invgamma.betas),
                         - 0.5 * (np.log(self.invgamma.betas) - digamma(self.invgamma.alphas)),
                         - 0.5 * self.invgamma.alphas / self.invgamma.betas])


class MatrixNormalInverseWishart(Distribution):

    def __init__(self, M, V, affine, psi, nu):
        self.matnorm = MatrixNormal(M=M, V=V)
        self.invwishart = InverseWishart(psi=psi, nu=nu)
        self.affine = affine

    @property
    def din(self):
        return self.matnorm.dcol

    @property
    def dout(self):
        return self.matnorm.drow

    @property
    def params(self):
        return self.matnorm.M, self.matnorm.V, self.invwishart.psi, self.invwishart.nu

    @params.setter
    def params(self, values):
        self.matnorm.M, self.matnorm.V, self.invwishart.psi, self.invwishart.nu = values

    def rvs(self, size=None):
        # sample sigma from inverse wishart
        sigma = self.invwishart.rvs()
        # sample mean from matrix-normal
        self.matnorm.U = sigma
        A = self.matnorm.rvs()
        return A, sigma

    def log_likelihood(self, x):
        A, sigma = x
        return MatrixNormal(M=self.matnorm.M, V=self.matnorm.V, U=sigma).log_likelihood(A)\
               + self.invwishart.log_likelihood(sigma)

    def mean(self):
        return tuple([self.matnorm.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.matnorm.mode(), self.invwishart.mode()])

    def log_partition(self):
        return 0.5 * self.invwishart.nu * self.dout * np.log(2) +\
               multigammaln(self.invwishart.nu / 2., self.dout) +\
               0.5 * self.dout * np.log(2. * np.pi) -\
               self.dout * np.sum(np.log(np.diag(self.matnorm.V_chol))) -\
               self.invwishart.nu * np.sum(np.log(np.diag(self.invwishart.psi_chol)))
        # n = self.dout
        # return n * self.invwishart.nu / 2 * np.log(2) + multigammaln(self.invwishart.nu / 2., n) \
        #        - self.invwishart.nu / 2 * np.linalg.slogdet(self.invwishart.psi)[1] - n / 2 * np.linalg.slogdet(self.matnorm.V)[1]

        # D = self.invwishart.psi.shape[0]
        # chol = self.invwishart.psi_chol
        # return -1 * (self.invwishart.nu * np.log(chol.diagonal()).sum() - (self.invwishartnu * D / 2 * np.log(2) + D * (D - 1) / 4 * np.log(np.pi) \
        #                                                    + gammaln((self.invwishart.nu - np.arange(D)) / 2).sum()))

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self._standard_to_nat(self.matnorm.M, self.matnorm.V,
                                     self.invwishart.psi, self.invwishart.nu)

    @nat_param.setter
    def nat_param(self, natparam):
        self.matnorm.M, self.matnorm.V,\
        self.invwishart.psi, self.invwishart.nu = self._nat_to_standard(natparam)

    @staticmethod
    def _standard_to_nat(M, V, psi, nu):
        V_inv = inv_psd(V)
        _psi = psi + M.dot(V_inv).dot(M.T)
        _M = M.dot(V_inv)
        _V = V_inv
        _nu = nu
        return np.array([_M, _V, _psi, _nu])

    @staticmethod
    def _nat_to_standard(natparam):
        # (yxT, xxT, yyT, n)
        nu = natparam[3]
        V = inv_psd(natparam[1])
        M = np.linalg.solve(natparam[1], natparam[0].T).T

        # This subtraction seems unstable!
        # It does not necessarily return a PSD matrix
        psi = natparam[2] - M.dot(natparam[0].T)

        # numerical padding here...
        V += 1e-8 * np.eye(V.shape[0])
        psi += 1e-8 * np.eye(psi.shape[0])
        assert np.all(0 < np.linalg.eigvalsh(psi))
        assert np.all(0 < np.linalg.eigvalsh(V))

        return M, V, psi, nu

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            data = data[~np.isnan(data).any(1)]
            n, dout, din = data.shape[0], self.dout, self.din

            stats = data.T.dot(data)
            xxT, yxT, yyT = stats[:-dout, :-dout], stats[-dout:, :-dout], stats[-dout:, -dout:]

            if self.affine:
                xy = np.sum(data, axis=0)
                x, y = xy[:-dout], xy[-dout:]
                xxT = blockarray([[xxT, x[:, np.newaxis]], [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return np.array([yxT, xxT, yyT, n])
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            gi = ~np.isnan(data).any(1)
            data, weights = data[gi], weights[gi]
            n, dout, din = weights.sum(), self.dout, self.din

            stats = data.T.dot(weights[:, np.newaxis] * data)
            xxT, yxT, yyT = stats[:-dout, :-dout], stats[-dout:, :-dout], stats[-dout:, -dout:]

            if self.affine:
                xy = weights.dot(data)
                x, y = xy[:-dout], xy[-dout:]
                xxT = blockarray([[xxT, x[:, np.newaxis]], [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return np.array([yxT, xxT, yyT, n])
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.dout, self.din)),
                         np.zeros((self.din, self.din)),
                         np.zeros((self.dout, self.dout)), 0])

    def get_expected_statistics(self):
        E_Sigmainv = self.invwishart.nu * np.linalg.inv(self.invwishart.psi)
        E_Sigmainv_A = self.invwishart.nu * np.linalg.solve(self.invwishart.psi, self.matnorm.M)
        E_AT_Sigmainv_A = self.dout * self.matnorm.V + self.invwishart.nu *\
                          self.matnorm.M.T.dot(np.linalg.solve(self.invwishart.psi, self.matnorm.M))
        E_logdetSigmainv = digamma((self.invwishart.nu - np.arange(self.dout)) / 2.).sum() +\
                           self.dout * np.log(2) - np.linalg.slogdet(self.invwishart.psi)[1]

        return E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv


class NormalInverseWishartMatrixNormalInverseWishart(Distribution):

        def __init__(self, mu, kappa, psi_niw, nu_niw, M, V, affine, psi_mniw, nu_mniw):
            self.gaussian = Gaussian(mu=mu)
            self.invwishart_niw = InverseWishart(psi=psi_niw, nu=nu_niw)
            self.kappa = kappa

            self.matnorm = MatrixNormal(M=M, V=V)
            self.invwishart_mniw = InverseWishart(psi=psi_mniw, nu=nu_mniw)
            self.affine = affine

        # def __init__(self, mu, kappa, psi, nu):
        #     self.gaussian = Gaussian(mu=mu)
        #     self.invwishart = InverseWishart(psi=psi, nu=nu)
        #     self.kappa = kappa

        @property
        def din(self):
            return self.matnorm.dcol

        @property
        def dout(self):
            return self.matnorm.drow

        @property
        def dim(self):
            return self.gaussian.dim

        @property
        def params(self):
            return self.gaussian.mu, self.kappa, self.invwishart_niw.psi, self.invwishart_niw.nu, \
                   self.matnorm.M, self.matnorm.V, self.invwishart_mniw.psi, self.invwishart_mniw.nu

        # @property
        # def params(self):
        #     return self.gaussian.mu, self.kappa, self.invwishart.psi, self.invwishart.nu

        @params.setter
        def params(self, values):
            self.gaussian.mu, self.kappa, self.invwishart_niw.psi, self.invwishart_niw.nu, \
            self.matnorm.M, self.matnorm.V, self.invwishart_mniw.psi, self.invwishart_mniw.nu = values

        # @params.setter
        # def params(self, values):
        #     self.gaussian.mu, self.kappa, self.invwishart.psi, self.invwishart.nu = values

        def rvs(self, size=None):
            # sample sigma from inverse wishart (niw)
            sigma_niw = self.invwishart_niw.rvs()
            # sample mean from gaussian
            self.gaussian.sigma = sigma_niw / self.kappa
            mu = self.gaussian.rvs()

            # sample sigma from inverse wishart (mniw)
            sigma_mniw = self.invwishart_mniw.rvs()
            # sample mean from matrix-normal
            self.matnorm.U = sigma_mniw
            A = self.matnorm.rvs()

            return mu, sigma_niw, A, sigma_mniw

        # def rvs(self, size=None):
        #     # sample sigma from inverse wishart
        #     sigma = self.invwishart.rvs()
        #     # sample mean from gaussian
        #     self.gaussian.sigma = sigma / self.kappa
        #     mu = self.gaussian.rvs()
        #     return mu, sigma

        def log_likelihood(self, x):
            mu, sigma_niw, A, sigma_mniw = x
            return Gaussian(mu=self.gaussian.mu, sigma=sigma_niw / self.kappa).log_likelihood(mu) + \
                   self.invwishart_niw.log_likelihood(sigma_niw) + \
                   MatrixNormal(M=self.matnorm.M, V=self.matnorm.V, U=sigma_mniw).log_likelihood(A) + \
                   self.invwishart_mniw.log_likelihood(sigma_mniw)

        # def log_likelihood(self, x):
        #     mu, sigma = x
        #     return Gaussian(mu=self.gaussian.mu, sigma=sigma / self.kappa).log_likelihood(mu) + \
        #            self.invwishart.log_likelihood(sigma)

        def mean(self):
            return tuple([self.gaussian.mean(), self.invwishart_niw.mean(), self.matnorm.mean(), self.invwishart_mniw.mean()])

        # def mean(self):
        #     return tuple([self.gaussian.mean(), self.invwishart.mean()])

        def mode(self):
            return tuple([self.gaussian.mode(), self.invwishart_niw.mode(), self.matnorm.mode(), self.invwishart_mniw.mode()])

        # def mode(self):
        #     return tuple([elf.gaussian.mode(), self.invwishart.mode()s])

        def log_partition(self):
            return 0.5 * self.invwishart_niw.nu * self.dim * np.log(2) + \
                   multigammaln(self.invwishart_niw.nu / 2., self.dim) + \
                   0.5 * self.dim * np.log(2. * np.pi / self.kappa) - \
                   self.invwishart_niw.nu * np.sum(np.log(np.diag(self.invwishart_niw.psi_chol))) + \
                   0.5 * self.invwishart_mniw.nu * self.dout * np.log(2) + \
                   multigammaln(self.invwishart_mniw.nu / 2., self.dout) + \
                   0.5 * self.dout * np.log(2. * np.pi) - \
                   self.dout * np.sum(np.log(np.diag(self.matnorm.V_chol))) - \
                   self.invwishart_mniw.nu * np.sum(np.log(np.diag(self.invwishart_mniw.psi_chol)))
            # n = self.dout
            # return 0.5 * self.invwishart_niw.nu * self.dim * np.log(2) + \
            #        multigammaln(self.invwishart_niw.nu / 2., self.dim) + \
            #        0.5 * self.dim * np.log(2. * np.pi / self.kappa) - \
            #        self.invwishart_niw.nu * np.sum(np.log(np.diag(self.invwishart_niw.psi_chol))) + \
            #        n * self.invwishart_mniw.nu / 2 * np.log(2) + multigammaln(self.invwishart_mniw.nu / 2., n) \
            #        - self.invwishart_mniw.nu / 2 * np.linalg.slogdet(self.invwishart_mniw.psi)[1] - n / 2 * np.linalg.slogdet(self.matnorm.V)[1]

        # def log_partition(self):
        #     return 0.5 * self.invwishart.nu * self.dim * np.log(2) + \
        #            multigammaln(self.invwishart.nu / 2., self.dim) + \
        #            0.5 * self.dim * np.log(2. * np.pi / self.kappa) - \
        #            self.invwishart.nu * np.sum(np.log(np.diag(self.invwishart.psi_chol)))

        def entropy(self):
            raise NotImplementedError

        # def entropy(self):
        #     raise NotImplementedError

        @property
        def nat_param(self):
            return self._standard_to_nat(self.gaussian.mu, self.kappa, self.invwishart_niw.psi, self.invwishart_niw.nu,
                                         self.matnorm.M, self.matnorm.V, self.invwishart_mniw.psi, self.invwishart_mniw.nu)
        # @property
        # def nat_param(self):
        #     return self._standard_to_nat(self.gaussian.mu, self.kappa,
        #                                  self.invwishart.psi, self.invwishart.nu)

        @nat_param.setter
        def nat_param(self, natparam):
            self.gaussian.mu, self.kappa, \
            self.invwishart_niw.psi, self.invwishart_niw.nu, \
            self.matnorm.M, self.matnorm.V, \
            self.invwishart_mniw.psi, self.invwishart_mniw.nu = self._nat_to_standard(natparam)

        # @nat_param.setter
        # def nat_param(self, natparam):
        #     self.gaussian.mu, self.kappa, \
        #     self.invwishart.psi, self.invwishart.nu = self._nat_to_standard(natparam)

        #@staticmethod
        def _standard_to_nat(self, mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw):
            _psi_niw = psi_niw + kappa * np.outer(mu, mu)
            _mu = kappa * mu
            _kappa = kappa
            _nu_niw = nu_niw + 2 + self.dim

            V_inv = inv_psd(V)
            _psi_mniw = psi_mniw + M.dot(V_inv).dot(M.T)
            _M = M.dot(V_inv)
            _V = V_inv
            _nu_mniw = nu_mniw

            return np.array([_mu, _kappa, _psi_niw, _nu_niw, _M, _V, _psi_mniw, _nu_mniw])

        # def _standard_to_nat(self, mu, kappa, psi, nu):
        #     _psi = psi + kappa * np.outer(mu, mu)
        #     _mu = kappa * mu
        #     _kappa = kappa
        #     _nu = nu + 2 + self.dim
        #     return np.array([_mu, _kappa, _psi, _nu])

        #@staticmethod
        def _nat_to_standard(self,natparam):
            kappa = natparam[1]
            mu = natparam[0] / kappa
            nu_niw = natparam[3] - 2 - self.dim
            psi_niw = natparam[2] - kappa * np.outer(mu, mu)

            # (yxT, xxT, yyT, n)
            nu_mniw = natparam[7]
            V = inv_psd(natparam[5])
            M = np.linalg.solve(natparam[5], natparam[4].T).T

            # This subtraction seems unstable!
            # It does not necessarily return a PSD matrix
            psi_mniw = natparam[6] - M.dot(natparam[4].T)

            # numerical padding here...
            V += 1e-8 * np.eye(V.shape[0])
            psi_mniw += 1e-8 * np.eye(psi_mniw.shape[0])
            assert np.all(0 < np.linalg.eigvalsh(psi_mniw))
            assert np.all(0 < np.linalg.eigvalsh(V))

            return mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw

        # def _nat_to_standard(self, natparam):
        #     kappa = natparam[1]
        #     mu = natparam[0] / kappa
        #     nu = natparam[3] - 2 - self.dim
        #     psi = natparam[2] - kappa * np.outer(mu, mu)
        #     return mu, kappa, psi, nu

        def get_statistics(self, data):
            if isinstance(data, np.ndarray):
                # data passed in like np.hstack((x, y))
                data = data[~np.isnan(data).any(1)]
                n_mniw, dout, din = data.shape[0], self.dout, self.din

                stats = data.T.dot(data)
                xxT_mniw, yxT_mniw, yyT_mniw = stats[:-dout, :-dout], stats[-dout:, :-dout], stats[-dout:, -dout:]

                if self.affine:
                    xy_mniw = np.sum(data, axis=0)
                    x_mniw, y_mniw = xy_mniw[:-dout], xy_mniw[-dout:]
                    xxT_mniw = blockarray([[xxT_mniw, x_mniw[:, np.newaxis]], [x_mniw[np.newaxis, :], np.atleast_2d(n_mniw)]])
                    yxT_mniw = np.hstack((yxT_mniw, y_mniw[:, np.newaxis]))

                # xxT_niw = np.einsum('nk,nh->kh', data, data)
                xxT_niw = stats[:-dout, :-dout]

                x_niw = np.sum(data[:,:-dout], axis=0)
                n_niw = data.shape[0]

                return np.array([x_niw, n_niw, xxT_niw, n_niw, yxT_mniw, xxT_mniw, yyT_mniw, n_mniw])
            else:
                return sum(list(map(self.get_statistics, data)), self._empty_statistics())

        # def get_statistics(self, data):
        #     if isinstance(data, np.ndarray):
        #         idx = ~np.isnan(data).any(1)
        #         data = data[idx]
        #
        #         xxT = np.einsum('nk,nh->kh', data, data)
        #         x = np.sum(data, axis=0)
        #         n = data.shape[0]
        #         return np.array([x, n, xxT, n])
        #     else:
        #         return sum(list(map(self.get_statistics, data)), self._empty_statistics())

        def get_weighted_statistics(self, data, weights):
            if isinstance(data, np.ndarray):
                idx = ~np.isnan(data).any(1)
                data = data[idx]
                weights = weights[idx]

                # data passed in like np.hstack((x, y))
                n_mniw, dout, din = weights.sum(), self.dout, self.din

                stats = data.T.dot(weights[:, np.newaxis] * data)
                xxT_mniw, yxT_mniw, yyT_mniw = stats[:-dout, :-dout], stats[-dout:, :-dout], stats[-dout:, -dout:]

                if self.affine:
                    xy = weights.dot(data)
                    x, y = xy[:-dout], xy[-dout:]
                    xxT_mniw = blockarray([[xxT_mniw, x[:, np.newaxis]], [x[np.newaxis, :], np.atleast_2d(n_mniw)]])
                    yxT_mniw = np.hstack((yxT_mniw, y[:, np.newaxis]))

                xxT_niw = stats[:-dout, :-dout]
                x_niw = weights.dot(data[:, :-dout])
                n_niw = weights.sum()

                return np.array([x_niw, n_niw, xxT_niw, n_niw, yxT_mniw, xxT_mniw, yyT_mniw, n_mniw])
            else:
                return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

        # def get_weighted_statistics(self, data, weights):
        #     if isinstance(data, np.ndarray):
        #         idx = ~np.isnan(data).any(1)
        #         data = data[idx]
        #         weights = weights[idx]
        #
        #         xxT = np.einsum('nk,n,nh->kh', data, weights, data)
        #         x = weights.dot(data)
        #         n = weights.sum()
        #         return np.array([x, n, xxT, n])
        #     else:
        #         return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

        def _empty_statistics(self):
            return np.array([np.zeros((self.dim,)), 0,
                             np.zeros((self.dim, self.dim)), 0,
                             np.zeros((self.dout, self.din)),
                             np.zeros((self.din, self.din)),
                             np.zeros((self.dout, self.dout)), 0])

        # def _empty_statistics(self):
        #     return np.array([np.zeros((self.dim,)), 0,
        #                      np.zeros((self.dim, self.dim)), 0])

        def get_expected_statistics(self):
            E_J = self.invwishart_niw.nu * np.linalg.inv(self.invwishart_niw.psi)
            E_h = self.invwishart_niw.nu * np.linalg.solve(self.invwishart_niw.psi, self.gaussian.mu)
            E_muJmuT = self.dim / self.kappa + self.gaussian.mu.dot(E_h)
            E_logdetSigmainv_niw = np.sum(digamma((self.invwishart_niw.nu - np.arange(self.dim)) / 2.)) + \
                               self.dim * np.log(2.) - np.linalg.slogdet(self.invwishart_niw.psi)[1]

            E_Sigmainv = self.invwishart_mniw.nu * np.linalg.inv(self.invwishart_mniw.psi)
            E_Sigmainv_A = self.invwishart_mniw.nu * np.linalg.solve(self.invwishart_mniw.psi, self.matnorm.M)
            E_AT_Sigmainv_A = self.dout * self.matnorm.V + self.invwishart_mniw.nu * \
                              self.matnorm.M.T.dot(np.linalg.solve(self.invwishart_mniw.psi, self.matnorm.M))
            E_logdetSigmainv_mniw = digamma((self.invwishart_mniw.nu - np.arange(self.dout)) / 2.).sum() + \
                               self.dout * np.log(2) - np.linalg.slogdet(self.invwishart_mniw.psi)[1]

            return E_J, E_h, E_muJmuT, E_logdetSigmainv_niw, E_Sigmainv, \
                   E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv_mniw

        # def get_expected_statistics(self):
        #     E_J = self.invwishart.nu * np.linalg.inv(self.invwishart.psi)
        #     E_h = self.invwishart.nu * np.linalg.solve(self.invwishart.psi, self.gaussian.mu)
        #     E_muJmuT = self.dim / self.kappa + self.gaussian.mu.dot(E_h)
        #     E_logdetSigmainv = np.sum(digamma((self.invwishart.nu - np.arange(self.dim)) / 2.)) + \
        #                        self.dim * np.log(2.) - np.linalg.slogdet(self.invwishart.psi)[1]
        #
        #     return E_J, E_h, E_muJmuT, E_logdetSigmainv