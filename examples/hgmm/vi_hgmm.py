import numpy as np
import numpy.random as npr

from mimo.distributions import Categorical
from mimo.distributions import TiedGaussiansWithPrecision

from mimo.mixtures import MixtureOfGaussians

import matplotlib.pyplot as plt

npr.seed(1337)

obs = []

# generate data
gating = Categorical(dim=2)

mus = np.stack([np.array([-10, -10.]),
                np.array([10., 10.])])

sigma = np.array([[3., 2.],
                  [2., 3.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='blue')

#
gating = Categorical(dim=2)

mus = np.stack([np.array([10, -10.]),
                np.array([-10., 10.])])

sigma = np.array([[3., -2.],
                  [-2., 3.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='red')

#
gating = Categorical(dim=2)

mus = np.stack([np.array([-10., 0.]),
                np.array([10., 0.])])

sigma = np.array([[5., 0.],
                  [0., 1.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='green')


gating = Categorical(dim=2)

mus = np.stack([np.array([0., -10.]),
                np.array([0., 10.])])

sigma = np.array([[1., 0.],
                  [0., 5.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='magenta')

plt.xlim((-15., 15.))
plt.ylim((-15., 15.))
plt.show()

obs = np.vstack(obs)

# learn model
from mimo.distributions import NormalWishart

from mimo.distributions import TiedGaussiansWithScaledPrecision
from mimo.distributions import TiedGaussiansWithHierarchicalNormalWisharts

from mimo.distributions import Dirichlet
from mimo.distributions import TruncatedStickBreaking
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfGaussiansWithHierarchicalPrior


cluster_size = 4
mixture_size = 2
dim = 2

# gating_prior = Dirichlet(dim=cluster_size, alphas=np.ones((cluster_size, )))
# gating = CategoricalWithDirichlet(dim=cluster_size, prior=gating_prior)

gating_prior = TruncatedStickBreaking(dim=cluster_size, gammas=np.ones((cluster_size, )),
                                      deltas=1. * np.ones((cluster_size,)))
gating = CategoricalWithStickBreaking(dim=cluster_size, prior=gating_prior)

components = []
for _ in range(cluster_size):
    # lower gating
    # _local_gating_prior = Dirichlet(dim=mixture_size, alphas=np.ones((mixture_size,)))
    # _local_gating = CategoricalWithDirichlet(dim=mixture_size, prior=_local_gating_prior)

    _local_gating_prior = TruncatedStickBreaking(dim=mixture_size, gammas=np.ones((mixture_size,)),
                                                 deltas=1. * np.ones((mixture_size,)))
    _local_gating = CategoricalWithStickBreaking(dim=mixture_size, prior=_local_gating_prior)

    # lower components
    _local_components_hyper_prior = NormalWishart(dim=dim,
                                                  mu=np.zeros((dim,)), kappa=1e-2,
                                                  psi=np.eye(dim), nu=dim + 1 + 1e-8)

    _local_components_prior = TiedGaussiansWithScaledPrecision(size=mixture_size, dim=dim,
                                                               kappas=1e-2 * np.ones((mixture_size,)))

    _local_components = TiedGaussiansWithHierarchicalNormalWisharts(size=mixture_size, dim=dim,
                                                                    hyper_prior=_local_components_hyper_prior,
                                                                    prior=_local_components_prior)

    _mixture = BayesianMixtureOfGaussiansWithHierarchicalPrior(mixture_size, dim,
                                                               gating=_local_gating,
                                                               components=_local_components)
    components.append(_mixture)


from mimo.mixtures import BayesianMixtureOfMixtureOfGaussians

model = BayesianMixtureOfMixtureOfGaussians(cluster_size, mixture_size, dim,
                                            gating=gating, components=components)

model.resample(obs, maxiter=25, maxsubiter=10, maxsubsubiter=5)

# model.meanfield_coordinate_descent(obs, maxiter=500, randomize=True,
#                                    maxsubiter=100, maxsubsubiter=25)

model.meanfield_stochastic_descent(obs, maxiter=500, randomize=True,
                                   maxsubiter=5, maxsubsubiter=5,
                                   step_size=1e-2, batch_size=64)

model.plot(obs)
