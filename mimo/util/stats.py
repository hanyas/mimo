import numpy as np
from numpy import random as npr

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


def multivariate_gaussian_loglik(xs, mu, lmbda, logdet_lmbda=None):
    # Accepts vectorized parameters
    d = mu.shape[-1]

    xc = np.nan_to_num(xs, copy=False) - mu
    log_exp = - 0.5 * np.einsum('...k,...kh,...h->...', xc, lmbda, xc)
    log_norm = - 0.5 * d * np.log(2. * np.pi)

    if logdet_lmbda is not None:
        log_norm += 0.5 * logdet_lmbda
    else:
        log_norm += 0.5 * np.linalg.slogdet(lmbda)[1]

    return log_norm + log_exp


def multivariate_studentt_loglik(xs, mu, lmbda, df):
    # Accepts vectorized parameters
    d = mu.shape[-1]

    xc = np.nan_to_num(xs, copy=False) - mu
    delta = np.einsum('...k,...kh,...h->...', xc, lmbda, xc)
    return sc.special.gammaln((df + d) / 2.) - sc.special.gammaln(df / 2.)\
           + 0.5 * np.linalg.slogdet(lmbda)[1] - (d / 2.) * np.log(df * np.pi)\
           - 0.5 * (df + d) * np.log1p(delta / df)
