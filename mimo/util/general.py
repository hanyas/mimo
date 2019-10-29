import numpy as np
import numpy.random as npr
import scipy.linalg.lapack as lapack
from scipy.special import logsumexp


def sample_discrete_from_log(p_log, return_lognorms=False, axis=0, dtype=np.int32):
    'samples log probability array along specified axis'
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
    # data is either an array (possibly rnd_V maskedarray) or rnd_V list of arrays
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
