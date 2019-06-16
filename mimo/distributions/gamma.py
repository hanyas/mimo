#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: wishart.py
# @Date: 2019-06-07-13-36
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import numpy as np
import numpy.random as npr

from scipy.special import gammaln, digamma

from mimo.abstractions import Distribution


class Gamma(Distribution):

    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def dim(self):
        return len(self.alphas)

    def rvs(self, size=None):
        # numpy uses a different parameterization
        return npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        return self.alphas / self.betas

    def mode(self):
        return np.where(self.alphas > 1., (self.alphas - 1.) / self.betas, 100000)

    def log_likelihood(self, x):
        loglik = np.sum(- gammaln(self.alphas) + self.alphas * np.log(self.betas) +
                          (self.alphas - 1.) * np.log(x) - x * self.betas)

        return loglik

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def entropy(self):
        return np.sum(self.alphas - np.log(self.betas) +
                      gammaln(self.alphas) + (1. - self.alphas) * digamma(self.alphas))


class InverseGamma(Distribution):

    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    @property
    def params(self):
        return self.alphas, self.betas

    @params.setter
    def params(self, values):
        self.alphas, self.betas = values

    @property
    def dim(self):
        return len(self.alphas)

    def rvs(self, size=None):
        # numpy uses a different parameterization
        return 1. / npr.gamma(self.alphas, 1. / self.betas)

    def mean(self):
        return np.where(self.alphas > 1., self.betas / (self.alphas - 1), 100000)

    def mode(self):
        return self.betas / (self.alphas + 1.)

    def log_likelihood(self, x):
        loglik = np.sum(- gammaln(self.alphas) + self.alphas * np.log(self.betas) -
                          (self.alphas + 1.) * np.log(x) - x / self.betas)

        return loglik

    def log_partition(self):
        return np.sum(gammaln(self.alphas) - self.alphas * np.log(self.betas))

    def entropy(self):
        return np.sum(self.alphas + np.log(self.betas) +
                      gammaln(self.alphas) - (1. + self.alphas) * digamma(self.alphas))


if __name__ == "__main__":
    pass
