import numpy as np
from sklearn.decomposition import PCA


def transform(mu, trans=None):
    if trans is None:
        return mu
    else:
        return trans.transform(mu)


def inverse_transform(mu, var=None, trans=None):
    if trans is None:
        if var is None:
            return mu
        else:
            return mu, var
    else:
        _mu = trans.inverse_transform(mu)
        if var is None:
            return _mu
        else:
            mat = np.sqrt(trans.explained_variance_[:, None]) * trans.components_\
                if isinstance(trans, PCA) else np.diag(np.sqrt(trans.var_))

            _diag = np.stack(list(map(np.diag, var)))
            _covar = np.einsum('kh,nhj,ji->nki', mat, _diag, mat.T)
            _var = np.vstack(list(map(np.diag, _covar)))

            return _mu, _var


def tofloat(x):
    return x[0] if len(x) == 1 else x


def tolist(x):
    return [x] if not isinstance(x, list) else x


def islist(*args):
    return all(isinstance(_arg, list) for _arg in args)


def extendlists(l):
    l = [[_l] if not isinstance(_l, list) else _l for _l in l]
    maxlen = max(map(len, l))
    return [_l + [_l[-1]] * (maxlen - len(_l)) for _l in l]


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
