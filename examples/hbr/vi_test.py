import numpy as np
import numpy.random as npr

from copy import deepcopy

from numpy.random import multivariate_normal as mvn

from mimo.distributions import MatrixNormalWishart
from mimo.distributions import MatrixNormalWithPrecision

# npr.seed(1337)

d, m = 1, 1
N, T = 25, 100

M = npr.uniform(-10., 10.) * np.ones((d, m))
V = npr.uniform(1e-3, 1e-1) * np.eye(d)

weight_dist = MatrixNormalWithPrecision(M=M, V=V,
                                        K=1e0 * np.eye(m))

W = []
for n in range(N):
    _W = weight_dist.rvs()
    W.append(_W)

X, Y = [], []
for n in range(N):
    _X = np.random.uniform(-10., 10., (m, T))
    _Y = W[n] @ _X + mvn(mean=np.zeros((d, )),
                         cov=np.linalg.inv(V), size=T).T
    X.append(_X)
    Y.append(_Y)

import matplotlib.pyplot as plt

plt.figure()
for n in range(N):
    plt.scatter(X[n].T, Y[n].T)
plt.show()

param_prior = MatrixNormalWishart(M=np.zeros((d, m)),
                                  K=1e-2 * np.eye(m),
                                  psi=np.eye(d), nu=2)

param_posterior = deepcopy(param_prior)

weight_prior = []
for n in range(N):
    _weight_prior = MatrixNormalWithPrecision(M=np.zeros((d, m)),
                                              V=np.eye(d),
                                              K=1e0 * np.eye(m))
    weight_prior.append(_weight_prior)

weight_posterior = deepcopy(weight_prior)


def update_weights(param_posterior, weight_prior,
                   weight_posterior, Y, X):

    S = param_posterior.matnorm.M
    psi = param_posterior.wishart.psi
    nu = param_posterior.wishart.nu

    for n in range(N):
        _X, _Y = X[n], Y[n]
        _K0 = weight_prior[n].K

        weight_posterior[n].K = (_K0 + _X @ _X.T)
        weight_posterior[n].M = (S @ _K0 + _Y @ _X.T) @ np.linalg.inv(weight_posterior[n].K)
        weight_posterior[n].V = nu * psi


def update_posterior(weight_posterior, weight_prior,
                     param_posterior, param_prior, Y, X):

    S0 = param_prior.matnorm.M
    R0 = param_prior.matnorm.K
    psi0 = param_prior.wishart.psi
    nu0 = param_prior.wishart.nu

    param_posterior.matnorm.M = S0 @ R0
    param_posterior.matnorm.K = deepcopy(R0)

    param_posterior.wishart.psi = np.linalg.inv(psi0) + S0.dot(R0).dot(S0.T)
    param_posterior.wishart.nu = deepcopy(nu0)

    for n in range(N):
        _X, _Y = X[n], Y[n]

        param_posterior.matnorm.M += weight_posterior[n].M @ weight_prior[n].K / N
        param_posterior.matnorm.K += weight_prior[n].K / N

        yyT = _Y @ _Y.T
        xxT = _X @ _X.T
        xyT = _X @ _Y.T

        _tr0 = np.trace(weight_prior[n].K @ np.linalg.inv(weight_posterior[n].K))
        mkmT = weight_posterior[n].M @ weight_prior[n].K @ weight_posterior[n].M.T

        _tr1 = np.trace(xxT @ np.linalg.inv(weight_posterior[n].K))
        mxxTmT = weight_posterior[n].M @ xxT @ weight_posterior[n].M.T
        mxyT = weight_posterior[n].M @ xyT

        param_posterior.wishart.nu += (T + m) / N
        param_posterior.wishart.psi += (mkmT + (_tr0 + _tr1) * np.linalg.inv(weight_posterior[n].V)
                                        + yyT - 2 * mxyT + mxxTmT) / N

    param_posterior.matnorm.M = param_posterior.matnorm.M @ np.linalg.inv(param_posterior.matnorm.K)
    param_posterior.wishart.psi = np.linalg.inv(param_posterior.wishart.psi
                                                - param_posterior.matnorm.M.dot(param_posterior.matnorm.K)
                                                .dot(param_posterior.matnorm.M.T))


def lower_bound(weight_prior, weight_posterior,
                param_prior, param_posterior):
    vlb = 0.

    for n in range(N):
        vlb += np.sum(param_posterior.expected_log_likelihood(Y[n].T, X[n].T, affine=False))

    vlb += param_posterior.entropy()
    vlb -= param_posterior.cross_entropy(param_prior)

    for n in range(N):
        vlb += weight_posterior[n].relative_entropy(weight_prior[n])

    return vlb


vlb = []
for i in range(100):
    update_weights(param_posterior, weight_prior,
                   weight_posterior, Y, X)

    update_posterior(weight_posterior, weight_prior,
                     param_posterior, param_prior, Y, X)

    vlb.append(lower_bound(weight_prior, weight_posterior,
                           param_prior, param_posterior))

warn = np.diff(vlb)
if np.any(warn) < 0.:
    print('Something is wrong, ELBO decreased')

plt.figure()
plt.plot(vlb)
plt.show()

# Test implementation
from mimo.distributions import HierarchicalLinearGaussianWithMatrixNormalWishart
from mimo.distributions import HierarchicalLinearGaussianWithSharedPrecision

YT = [_y.T for _y in Y]
XT = [_x.T for _x in X]

likelihood = HierarchicalLinearGaussianWithSharedPrecision(M=np.zeros((d, m)),
                                                           K=1e0 * np.eye(m),
                                                           V=np.eye(d),
                                                           affine=False)
prior = MatrixNormalWishart(M=np.zeros((d, m)),
                            K=1e-2 * np.eye(m),
                            psi=np.eye(d), nu=2)

gibbs = HierarchicalLinearGaussianWithMatrixNormalWishart(prior=prior,
                                                          likelihood=likelihood)
gibbs.resample(YT, XT)


vi = HierarchicalLinearGaussianWithMatrixNormalWishart(prior=prior,
                                                       likelihood=likelihood)
vi.meanfield_update(YT, XT)

print('True Mean:', M)
print('True Precision:', V)

print('Mean of Matrix Normal:', param_posterior.matnorm.mean())
print('Mean of Wishart:', param_posterior.wishart.mean())

print('Gibbs: Mean of Matrix Normal:', gibbs.posterior.matnorm.mean())
print('Gibbs: Mean of Wishart:', gibbs.posterior.wishart.mean())

print('VI: Mean of Matrix Normal:', vi.posterior.matnorm.mean())
print('VI: Mean of Wishart:', vi.posterior.wishart.mean())
