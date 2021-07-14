import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import islice
import random


def batches(batchsize, datasize):
    idx_all = random.sample(range(datasize), batchsize)
    idx_iter = iter(idx_all)
    yield from iter(lambda: list(islice(idx_iter, batchsize)), [])


def transform(mu, trans=None):
    if trans is None:
        return mu
    else:
        return trans.transform(mu)


def inverse_transform_mean(mu, trans):
    if trans is None:
        return mu
    else:
        return trans.inverse_transform(mu)


def inverse_transform_variance(var, trans):
    if trans is None:
        return var
    else:
        mat = None
        if isinstance(trans, PCA):
            mat = np.sqrt(trans.explained_variance_[:, None]) * trans.components_
        elif isinstance(trans, StandardScaler):
            mat = np.diag(np.sqrt(trans.var_))
        elif isinstance(trans, MinMaxScaler):
            mat = np.diag(trans.scale_)

        return np.einsum('kh,...hj,ji->...ki', mat, var, mat.T)


def inverse_transform(mu, var, trans=None):
    _mu = inverse_transform_mean(mu, trans)
    _var = inverse_transform_variance(var, trans)
    return _mu, _var


def tofloat(x):
    return x[0] if len(x) == 1 else x


def tolist(x):
    return [x] if not isinstance(x, list) else x


def islist(*args):
    return all(isinstance(_arg, list) for _arg in args)


def anynone(*args):
    return any(_ is None for _ in args)


def atleast2d(data):
    if data.ndim == 1:
        return data.reshape((-1, 1))
    return data


def gi(data):
    out = (np.isnan(atleast2d(data)).sum(1) == 0).ravel()
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


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    N, shp = z.size, z.shape

    zoh = np.zeros((K, N))
    zoh[np.arange(K)[np.ravel(z)], np.arange(N)] = 1
    zoh = np.reshape(zoh, (K,) + shp)

    return zoh
