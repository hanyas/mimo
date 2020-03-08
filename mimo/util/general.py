import numpy as np
import numpy.random as npr

import scipy as sc
import scipy.linalg.lapack as lapack
from scipy.special import logsumexp

from numpy.core.umath_tests import inner1d


def is_pd(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def near_pd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


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


def multivariate_t_loglik(y, mu, nu, lmbda):
    # returns the log value
    d = len(mu)
    yc = np.array(y - mu, ndmin=2)
    L = np.linalg.cholesky(lmbda)
    ys = sc.linalg.solve_triangular(L, yc.T, overwrite_b=True, lower=True)
    return sc.special.gammaln((nu + d) / 2.) - sc.special.gammaln(nu / 2.) \
            - (d / 2.) * np.log(nu * np.pi) - np.log(L.diagonal()).sum() \
            - (nu + d) / 2. * np.log1p(1. / nu * inner1d(ys.T, ys.T))


def matrix_studentt(x, M, V, psi, nu, affine=True):
    if affine:
        x = np.hstack((x, 1.))

    xxT = np.outer(x, x)

    # https://tminka.github.io/papers/minka-linear.pdf
    c = 1. - x.T @ np.linalg.inv(np.linalg.inv(V) + xxT) @ x

    # https://tminka.github.io/papers/minka-gaussian.pdf
    df = nu
    mu = M @ x
    sigma = (1. / c) * psi / df  # Misleading in Minka
    var = sigma * df / (df - 2)

    # # variance of approximate Gaussian
    # var = psi / df

    return mu, var, df


# data
def any_none(*args):
    return any(_ is None for _ in args)


def atleast_2d(data):
    # NOTE: can't use np.atleast_2d because if it's 1D we want axis 1 to be the
    # singleton and axis 0 to be the sequence index
    if data.ndim == 1:
        return data.reshape((-1, 1))
    return data


def gi(data):
    out = (np.isnan(atleast_2d(data)).sum(1) == 0).ravel()
    return out if len(out) != 0 else None


def normalizedata(data, scaling):
    # Normalize data to 0 mean, 1 std_deviation, optionally scale data
    mean = np.mean(data, axis=0)
    std_deviation = np.std(data, axis=0)
    data = (data - mean) / (std_deviation * scaling)
    return data


def centerdata(data, scaling):
    # Center data to 0 mean
    mean = np.mean(data, axis=0)
    data = (data - mean) / scaling
    return data


def getdatasize(data):
    if isinstance(data, np.ma.masked_array):
        return data.shape[0] - data.mask.reshape((data.shape[0], -1))[:, 0].sum()
    elif isinstance(data, np.ndarray):
        if len(data) == 0:
            return 0
        return data[gi(data)].shape[0]
    elif isinstance(data, list):
        return sum(getdatasize(d) for d in data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data, int) or isinstance(data, float)
        return 1


def getdatadim(data):
    if isinstance(data, np.ndarray):
        assert data.ndim > 1
        return data.shape[1]
    elif isinstance(data, list):
        assert len(data) > 0
        return getdatadim(data[0])
    else:
        # handle unboxed case for convenience
        assert isinstance(data, int) or isinstance(data, float)
    return 1


def combinedata(datas):
    ret = []
    for data in datas:
        if isinstance(data, np.ma.masked_array):
            ret.append(np.ma.compress_rows(data))
        if isinstance(data, np.ndarray):
            ret.append(data)
        elif isinstance(data, list):
            ret.extend(combinedata(data))
        else:
            # handle unboxed case for convenience
            assert isinstance(data, int) or isinstance(data, float)
            ret.append(np.atleast_1d(data))
    return ret


def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list) or isinstance(data, tuple):
        if any(isinstance(d, np.ma.MaskedArray) for d in data):
            return np.concatenate([np.ma.compress_rows(d) for d in data])
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data, int) or isinstance(data, float)
        return np.atleast_1d(data)


def cumsum(v, strict=False):
    if not strict:
        return np.cumsum(v, axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1], axis=0)
    return out


# matrix
def blockarray(*args, **kwargs):
    return np.array(np.bmat(*args, **kwargs), copy=False)


def copy_lower_to_upper(A):
    A += np.tril(A, k=-1).T


def inv_psd(A, return_chol=False):
    L = np.linalg.cholesky(A)
    Ainv = lapack.dpotri(L, lower=True)[0]
    copy_lower_to_upper(Ainv)
    if return_chol:
        return Ainv, L
    else:
        return Ainv


def sample_env(env, nb_rollouts, nb_steps,
               ctl=None, noise_std=0.1,
               apply_limit=True):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    ulim = env.action_space.high

    for n in range(nb_rollouts):
        _obs = np.zeros((nb_steps, dm_obs))
        _act = np.zeros((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                # unifrom distribution
                u = np.random.uniform(-ulim, ulim)
            else:
                u = ctl(x)
                u = u + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act
