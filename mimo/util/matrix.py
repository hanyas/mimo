import numpy as np
from scipy.linalg import lapack as lapack


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


def copy_lower_to_upper(A):
    A += np.tril(A, k=-1).T


def inv_pd(A, return_chol=False):
    L = np.linalg.cholesky(A)
    Ainv = lapack.dpotri(L, lower=True)[0]
    copy_lower_to_upper(Ainv)
    if return_chol:
        return Ainv, L
    else:
        return Ainv


def blockarray(*args, **kwargs):
    return np.array(np.bmat(*args, **kwargs), copy=False)


def symmetrize(A):
    return (A + A.T) / 2.
