import numpy as np
from numpy import random as npr

from numpy.core._umath_tests import inner1d

import scipy as sc
from scipy.special import logsumexp


def sample_discrete_from_log(p_log, return_lognorms=False, axis=0, dtype=np.int32):
    # samples log probability array along specified axis
    lognorms = logsumexp(p_log, axis=axis)
    cumvals = np.exp(p_log - np.expand_dims(lognorms, axis)).cumsum(axis)
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = npr.random(size=thesize) * \
               np.reshape(cumvals[tuple([slice(None) if i is not axis else -1
                                         for i in range(p_log.ndim)])], thesize)
    samples = np.sum(randvals > cumvals, axis=axis, dtype=dtype)
    if return_lognorms:
        return samples, lognorms
    else:
        return samples


def multivariate_studentt_loglik(y, mu, scale, nu):
    # Following Bishop notation
    d = len(mu)
    yc = np.array(y - mu, ndmin=2)
    L = np.linalg.cholesky(scale)
    ys = sc.linalg.solve_triangular(L, yc.T, overwrite_b=True, lower=True)
    return sc.special.gammaln((nu + d) / 2.) - sc.special.gammaln(nu / 2.) \
            - (d / 2.) * np.log(nu * np.pi) - np.sum(np.log(np.diag(L))) \
            - (nu + d) / 2. * np.log1p(1. / nu * inner1d(ys.T, ys.T))


def multivariate_gaussian_loglik(y, mu, scale):
    d = len(mu)
    yc = np.nan_to_num(y).reshape((-1, d)) - mu
    L = np.linalg.cholesky(scale)
    ys = sc.linalg.solve_triangular(L, yc.T, overwrite_b=True, lower=True)
    return - 0.5 * d * np.log(2. * np.pi)\
           - np.sum(np.log(np.diag(L))) - 0.5 * inner1d(ys.T, ys.T)


def matrix_linear_gaussian(x, M, V, psi, nu, affine=True):
    if affine:
        x = np.hstack((x, 1.))

    # https://tminka.github.io/papers/minka-gaussian.pdf
    mu = M @ x

    # variance of approximate Gaussian
    sigma = psi / nu  # Misleading in Minka

    return mu, sigma, nu


def matrix_linear_studentt(x, M, V, psi, nu, affine=True):
    if affine:
        x = np.hstack((x, 1.))

    xxT = np.outer(x, x)

    # https://tminka.github.io/papers/minka-linear.pdf
    c = 1. - x.T @ np.linalg.inv(np.linalg.inv(V) + xxT) @ x

    # https://tminka.github.io/papers/minka-gaussian.pdf
    df = nu
    mu = M @ x

    # variance of a student-t
    sigma = (1. / c) * psi / df  # Misleading in Minka
    var = sigma * df / (df - 2)

    return mu, sigma, df
