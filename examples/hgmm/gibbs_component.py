import numpy as np
import numpy.random as npr

from mimo.distributions import NormalWishart
from mimo.distributions import StackedGaussiansWithPrecision

from mimo.distributions import TiedGaussiansWithKnownScaledPrecision
from mimo.distributions import TiedGaussiansWithHierarchicalNormalWisharts

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import MixtureOfGaussians
from mimo.mixtures import BayesianMixtureOfGaussiansWithHierarchicalPrior


# npr.seed(1337)

# generate data
gating = Categorical(dim=4)

mus = np.stack([np.array([-3., 3.]),
                np.array([3., -3.]),
                np.array([5., 5.]),
                np.array([-5., -5.])])

lmbdas = np.stack([1. * np.eye(2),
                   1. * np.eye(2),
                   1. * np.eye(2),
                   1. * np.eye(2)])

components = StackedGaussiansWithPrecision(size=4, dim=2,
                                           mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, labels = gmm.rvs(500)
gmm.plot(obs)

# learn model
gating_prior = Dirichlet(dim=4, alphas=np.ones((4, )))

gating = CategoricalWithDirichlet(dim=4, prior=gating_prior)

mu = np.zeros((2, ))
kappa = 1e-2
psi = np.eye(2)
nu = 3. + 1e-8

components_hyper_prior = NormalWishart(dim=2,
                                       mu=mu, kappa=kappa,
                                       psi=psi, nu=nu)

etas = 1e-2 * np.ones((4,))
components_prior = TiedGaussiansWithKnownScaledPrecision(size=4, dim=2,
                                                         kappas=etas)

components = TiedGaussiansWithHierarchicalNormalWisharts(size=4, dim=2,
                                                         hyper_prior=components_hyper_prior,
                                                         prior=components_prior)

model = BayesianMixtureOfGaussiansWithHierarchicalPrior(gating=gating, components=components)

model.resample(obs, maxiter=1000, maxsubiter=1)

model.plot(obs)
