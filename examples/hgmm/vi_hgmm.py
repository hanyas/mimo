import numpy as np
import numpy.random as npr

from mimo.distributions import Categorical
from mimo.distributions import TiedGaussiansWithPrecision

from mimo.mixtures import MixtureOfGaussians

import matplotlib.pyplot as plt

# npr.seed(1337)

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

_obs, _ = gmm.rvs(250)
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

_obs, _ = gmm.rvs(250)
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

_obs, _ = gmm.rvs(250)
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

_obs, _ = gmm.rvs(250)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='magenta')

plt.xlim((-15., 15.))
plt.ylim((-15., 15.))
plt.show()


obs = np.vstack(obs)

# learn model
from mimo.distributions import NormalWishart

from mimo.distributions import TiedGaussiansWithKnownScaledPrecision
from mimo.distributions import TiedGaussiansWithHierarchicalNormalWisharts

from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import BayesianMixtureOfGaussiansWithHierarchicalPrior


upper_size = 8
lower_size = 4
dim = 2

gating_prior = Dirichlet(dim=upper_size, alphas=0.01 * np.ones((upper_size, )))
gating = CategoricalWithDirichlet(dim=upper_size, prior=gating_prior)

components = []
for _ in range(upper_size):
    # lower gating
    _local_gating_prior = Dirichlet(dim=lower_size, alphas=0.01 * np.ones((lower_size,)))
    _local_gating = CategoricalWithDirichlet(dim=lower_size, prior=_local_gating_prior)

    # lower components
    _mu, _kappa = np.zeros((dim,)), 1e-2
    _psi, _nu = np.eye(dim), 3. + 1e-32

    _local_components_hyper_prior = NormalWishart(dim=dim,
                                                  mu=_mu, kappa=_kappa,
                                                  psi=_psi, nu=_nu)

    _etas = 1e-2 * np.ones((lower_size,))
    _local_components_prior = TiedGaussiansWithKnownScaledPrecision(size=lower_size, dim=dim,
                                                                    kappas=_etas)

    _local_components = TiedGaussiansWithHierarchicalNormalWisharts(size=lower_size, dim=dim,
                                                                    hyper_prior=_local_components_hyper_prior,
                                                                    prior=_local_components_prior)

    _local_model = BayesianMixtureOfGaussiansWithHierarchicalPrior(gating=_local_gating,
                                                                   components=_local_components)
    components.append(_local_model)


from mimo.mixtures import BayesianMixtureOfMixtureOfGaussians

model = BayesianMixtureOfMixtureOfGaussians(gating=gating, components=components)

model.resample(obs, maxiter=10, maxsubiter=500, maxsubsubiter=1)
model.meanfield_coordinate_descent(obs, maxiter=100,
                                   maxsubiter=50, maxsubsubiter=25)

model.plot(obs)
