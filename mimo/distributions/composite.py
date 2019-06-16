#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: compose.py
# @Date: 2019-06-08-21-53
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


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
        return Gaussian(mu=self.gaussian.mu, sigma=sigma / self.kappa).log_likelihood(mu) +\
               self.invwishart.log_likelihood(sigma)

    def mean(self):
        return tuple([self.gaussian.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invwishart.mode()])

    def log_partition(self):
        return 0.5 * self.invwishart.nu * self.dim * np.log(2) +\
               multigammaln(self.nu / 2., self.dim) +\
               0.5 * self.dim * np.log(2. * np.pi / self.kappa) -\
               self.invwishart.nu * np.sum(np.log(np.diag(self.invwishart.psi_chol)))

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
        E_h = self.invwishart.nu * np.linalg.solve(self.psi, self.gaussian.mu)
        E_muJmuT = self.dim / self.kappa + self.gaussian.mu.dot(E_h)
        E_logdetSigmainv = np.sum(digamma((self.invwishart.nu - np.arange(self.dim)) / 2.)) +\
                           self.dim * np.log(2.) - np.linalg.slogdet(self.invwishart.psi)[1]

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
        return DiagonalGaussian(mu=self.gaussian.mu, sigmas=sigmas/self.kappa).log_likelihood(mu) +\
               self.invgamma.log_likelihood(sigmas)

    def mean(self):
        return tuple([self.gaussian.mean(), self.invgamma.mean()])

    def mode(self):
        return tuple([self.gaussian.mode(), self.invgamma.mode()])

    def log_partition(self):
        return np.sum(gammaln(self.invgamma.alphas) -
                      self.invgamma.alphas * np.log(self.invgamma.betas)) +\
               np.sum(0.5 * np.log(2. * np.pi / self.kappas))

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
        return MatrixNormal(M=self.matnorm.M, V=self.matnorm.V, U=sigma).log_likelihood(A) +\
               self.invwishart.log_likelihood(sigma)

    def mean(self):
        return tuple([self.matnorm.mean(), self.invwishart.mean()])

    def mode(self):
        return tuple([self.matnorm.mode(), self.invwishart.mode()])

    def log_partition(self):
        return 0.5 * self.invwishart.nu * self.dout * np.log(2) +\
               multigammaln(self.nu / 2., self.dout) +\
               0.5 * self.dout * np.log(2. * np.pi) -\
               self.dout * np.sum(np.log(np.diag(self.matnorm.V_chol))) -\
               self.invwishart.nu * np.sum(np.log(np.diag(self.invwishart.psi_chol)))

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
