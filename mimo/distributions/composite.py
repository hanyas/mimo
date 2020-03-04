import numpy as np

from mimo.abstractions import Distribution
from mimo.distributions import InverseWishart
from mimo.distributions import InverseGamma
from mimo.distributions import Gaussian, DiagonalGaussian
from mimo.distributions import MatrixNormal

from scipy.special import multigammaln, digamma, gammaln

from mimo.util.general import blockarray, inv_psd, near_pd


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

    def mean(self):
        return tuple([self.gaussian.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invwishart.mode()])

    def log_likelihood(self, x):
        mu, sigma = x
        return Gaussian(mu=self.gaussian.mu, sigma=sigma / self.kappa).log_likelihood(mu)\
               + self.invwishart.log_likelihood(sigma)

    def log_partition(self, params=None):
        mu, kappa, psi, nu = params if params is not None else self.params
        return 0.5 * nu * self.dim * np.log(2)\
               + multigammaln(nu / 2., self.dim)\
               + 0.5 * self.dim * np.log(2. * np.pi / kappa)\
               - 0.5 * nu * np.linalg.slogdet(near_pd(psi))[1]

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.standard_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_standard(natparam)

    @staticmethod
    def standard_to_nat(params):
        psi = params[2] + params[1] * np.outer(params[0], params[0])
        mu = params[1] * params[0]
        kappa = params[1]
        nu = params[3] + 2 + params[0].shape[0]
        return np.array([mu, kappa, psi, nu])

    @staticmethod
    def nat_to_standard(natparam):
        kappa = natparam[1]
        mu = natparam[0] / kappa
        nu = natparam[3] - 2 - mu.shape[0]
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

    def mean(self):
        return tuple([self.gaussian.mean(), self.invgamma.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invgamma.mode()])

    def log_likelihood(self, x):
        mu, sigmas = x
        return DiagonalGaussian(mu=self.gaussian.mu, sigmas=sigmas/self.kappa).log_likelihood(mu)\
               + self.invgamma.log_likelihood(sigmas)

    def log_partition(self, params=None):
        mu, kappas, alphas, betas = params if params is not None else self.params
        return np.sum(gammaln(alphas) - alphas * np.log(betas))\
               + np.sum(0.5 * np.log(2. * np.pi / kappas))

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.standard_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_standard(natparam)

    @staticmethod
    def standard_to_nat(params):
        mu, kappas, alphas, betas = params
        return np.array([kappas * mu, kappas, 2. * alphas, 2. * betas + kappas * mu**2])

    @staticmethod
    def nat_to_standard(natparam):
        kappas = natparam[1]
        mu = natparam[0] / kappas
        alphas = natparam[2] / 2.
        betas = (natparam[3] - kappas * mu**2) / 2.
        return mu, kappas, alphas, betas

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

    def rvs(self, size=None):
        # sample sigma from inverse wishart
        sigma = self.invwishart.rvs()
        # sample mean from matrix-normal
        self.matnorm.U = sigma
        A = self.matnorm.rvs()
        return A, sigma

    def mean(self):
        return tuple([self.matnorm.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.matnorm.mode(), self.invwishart.mode()])

    def log_likelihood(self, x):
        A, sigma = x
        return MatrixNormal(M=self.matnorm.M, V=self.matnorm.V, U=sigma).log_likelihood(A)\
               + self.invwishart.log_likelihood(sigma)

    def log_partition(self, params=None):
        M, V, psi, nu = params if params is not None else self.params
        return 0.5 * nu * self.drow * np.log(2)\
               + multigammaln(nu / 2., self.drow)\
               + 0.5 * self.drow * np.log(2. * np.pi)\
               - 0.5 * self.drow * np.linalg.slogdet(near_pd(V))[1]\
               - 0.5 * nu * np.linalg.slogdet(near_pd(psi))[1]

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.standard_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_standard(natparam)

    @staticmethod
    def standard_to_nat(params):
        Vinv = inv_psd(params[1])
        psi = params[2] + params[0].dot(Vinv).dot(params[0].T)
        M = params[0].dot(Vinv)
        V = Vinv
        nu = params[3]
        return np.array([M, V, psi, nu])

    @staticmethod
    def nat_to_standard(natparam):
        # (yxT, xxT, yyT, n)
        nu = natparam[3]
        V = inv_psd(natparam[1])
        M = np.linalg.solve(natparam[1], natparam[0].T).T

        # This subtraction seems unstable!
        # It does not necessarily return a PSD matrix
        psi = natparam[2] - M.dot(natparam[0].T)

        # numerical paddcolg here...
        V = near_pd(V + 1e-8 * np.eye(V.shape[0]))
        psi = near_pd(psi + 1e-8 * np.eye(psi.shape[0]))

        assert np.all(0 < np.linalg.eigvalsh(psi))
        assert np.all(0 < np.linalg.eigvalsh(V))

        return M, V, psi, nu

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            data = data[~np.isnan(data).any(1)]
            n, drow, dcol = data.shape[0], self.drow, self.dcol

            stats = data.T.dot(data)
            xxT, yxT, yyT = stats[:-drow, :-drow], stats[-drow:, :-drow], stats[-drow:, -drow:]

            if self.affine:
                xy = np.sum(data, axis=0)
                x, y = xy[:-drow], xy[-drow:]
                xxT = blockarray([[xxT, x[:, np.newaxis]],
                                  [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return np.array([yxT, xxT, yyT, n])
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            gi = ~np.isnan(data).any(1)
            data, weights = data[gi], weights[gi]
            n, drow, dcol = weights.sum(), self.drow, self.dcol

            stats = data.T.dot(weights[:, np.newaxis] * data)
            xxT, yxT, yyT = stats[:-drow, :-drow], stats[-drow:, :-drow], stats[-drow:, -drow:]

            if self.affine:
                xy = weights.dot(data)
                x, y = xy[:-drow], xy[-drow:]
                xxT = blockarray([[xxT, x[:, np.newaxis]], [x[np.newaxis, :], np.atleast_2d(n)]])
                yxT = np.hstack((yxT, y[:, np.newaxis]))

            return np.array([yxT, xxT, yyT, n])
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.drow, self.dcol)),
                         np.zeros((self.dcol, self.dcol)),
                         np.zeros((self.drow, self.drow)), 0])

    def get_expected_statistics(self):
        E_Sigmainv = self.invwishart.nu * np.linalg.inv(self.invwishart.psi)
        E_Sigmainv_A = self.invwishart.nu * np.linalg.solve(self.invwishart.psi, self.matnorm.M)
        E_AT_Sigmainv_A = self.drow * self.matnorm.V + self.invwishart.nu\
                          * self.matnorm.M.T.dot(np.linalg.solve(self.invwishart.psi, self.matnorm.M))
        E_logdetSigmainv = digamma((self.invwishart.nu - np.arange(self.drow)) / 2.).sum() \
                           + self.drow * np.log(2) - np.linalg.slogdet(self.invwishart.psi)[1]

        return E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv


class NormalInverseWishartMatrixNormalInverseWishart(Distribution):

    def __init__(self, mu, kappa, psi_niw, nu_niw, M, V, affine, psi_mniw, nu_mniw):
        self.niw = NormalInverseWishart(mu=mu, kappa=kappa, psi=psi_niw, nu=nu_niw)
        self.mniw = MatrixNormalInverseWishart(M=M, V=V, affine=affine, psi=psi_mniw, nu=nu_mniw)

    @property
    def dcol(self):
        return self.mniw.matnorm.dcol

    @property
    def drow(self):
        return self.mniw.matnorm.drow

    @property
    def dim(self):
        return self.niw.gaussian.dim

    @property
    def params(self):
        return tuple([*self.niw.params, *self.mniw.params])

    @params.setter
    def params(self, values):
        self.niw.params = values[:4]
        self.mniw.params = values[4:]

    def rvs(self, size=None):
        # sample mu, sigma from normal inverse wishart (niw)
        mu, sigma_niw = self.niw.rvs()
        # sample A, sigma from matrix inverse wishart (mniw)
        A, sigma_mniw = self.mniw.rvs()
        return mu, sigma_niw, A, sigma_mniw

    def mean(self):
        return tuple([*self.niw.mean(), *self.mniw.mean()])

    def mode(self):
        return tuple([*self.niw.mode(), *self.mniw.mode()])

    def log_likelihood(self, x):
        return self.niw.log_likelihood(x[:2]) + self.mniw.log_likelihood(x[2:])

    def log_partition(self, params=None):
        if params is not None:
            niw_params, mniw_params = params[:4], params[4:]
        else:
            niw_params, mniw_params = self.params[:4], self.params[4:]
        return self.niw.log_partition(niw_params) + self.mniw.log_partition(mniw_params)

    def entropy(self):
        raise NotImplementedError

    @property
    def nat_param(self):
        return self.standard_to_nat(self.params)

    @nat_param.setter
    def nat_param(self, natparam):
        self.params = self.nat_to_standard(natparam)

    def standard_to_nat(self, params):
        niw_params = self.niw.standard_to_nat(params[:4])
        mniw_params = self.mniw.standard_to_nat(params[4:])
        return np.hstack((niw_params, mniw_params))

    def nat_to_standard(self, natparam):
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

        # numerical paddcolg here...
        psi_niw = near_pd(psi_niw + 1e-8 * np.eye(psi_niw.shape[0]))
        V = near_pd(V + 1e-8 * np.eye(V.shape[0]))
        psi_mniw = near_pd(psi_mniw + 1e-8 * np.eye(psi_mniw.shape[0]))

        assert np.all(0 < np.linalg.eigvalsh(psi_niw))
        assert np.all(0 < np.linalg.eigvalsh(psi_mniw))
        assert np.all(0 < np.linalg.eigvalsh(V))

        return mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw

    def get_statistics(self, data):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            drow, dcol = self.drow, self.dcol

            x, xy = data[:, :-drow], data

            niw_stats = self.niw.get_statistics(x)
            mniw_stats = self.mniw.get_statistics(xy)

            return np.hstack((niw_stats, mniw_stats))
        else:
            return sum(list(map(self.get_statistics, data)), self._empty_statistics())

    def get_weighted_statistics(self, data, weights):
        if isinstance(data, np.ndarray):
            # data passed in like np.hstack((x, y))
            drow, dcol = self.drow, self.dcol
            x, xy = data[:, :-drow], data

            niw_stats = self.niw.get_weighted_statistics(x, weights)
            mniw_stats = self.mniw.get_weighted_statistics(xy, weights)

            return np.hstack((niw_stats, mniw_stats))
        else:
            return sum(list(map(self.get_weighted_statistics, data, weights)), self._empty_statistics())

    def _empty_statistics(self):
        return np.array([np.zeros((self.dim,)), 0,
                         np.zeros((self.dim, self.dim)), 0,
                         np.zeros((self.drow, self.dcol)),
                         np.zeros((self.dcol, self.dcol)),
                         np.zeros((self.drow, self.drow)), 0])

    def get_expected_statistics(self):
        niw_stats = self.niw.get_expected_statistics()
        mniw_stats = self.mniw.get_expected_statistics()
        return tuple([*niw_stats, *mniw_stats])
