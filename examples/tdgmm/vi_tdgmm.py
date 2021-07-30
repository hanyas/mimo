import numpy as np
import numpy.random as npr

from mimo.distributions import StackedGaussiansWithPrecision
from mimo.distributions import TiedGaussiansWithNormalGammas
from mimo.distributions import TiedNormalGammas

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import MixtureOfGaussians
from mimo.mixtures import BayesianMixtureOfGaussians

from matplotlib import pyplot as plt


# npr.seed(1337)

# generate data
gating = Categorical(dim=4)

mus = np.stack([np.array([-3., 3.]),
                np.array([3., -3.]),
                np.array([5., 5.]),
                np.array([-5., -5.])])

lmbdas = np.stack([4. * np.eye(2),
                   3. * np.eye(2),
                   2. * np.eye(2),
                   1. * np.eye(2)])

components = StackedGaussiansWithPrecision(size=4, dim=2,
                                           mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, labels = gmm.rvs(500)
gmm.plot(obs)

# learn model
gating_prior = Dirichlet(dim=4, alphas=np.ones((4, )))

gating = CategoricalWithDirichlet(dim=4, prior=gating_prior)

mus = np.zeros((4, 2))
kappas = 1e-2 * np.ones((4, 2))
alphas = (3. + 1e-8) / 2. * np.ones((4, 2))
betas = 1. / (2. * np.ones((4, 2)))

components_prior = TiedNormalGammas(size=4, dim=2,
                                    mus=mus, kappas=kappas,
                                    alphas=alphas, betas=betas)

components = TiedGaussiansWithNormalGammas(size=4, dim=2,
                                           prior=components_prior)

model = BayesianMixtureOfGaussians(gating=gating, components=components)

vlb = model.meanfield_coordinate_descent(obs=obs, maxiter=5000, tol=0.)
print("vlb monoton?", np.all(np.diff(vlb) >= -1e-8))

plt.figure()
model.plot(obs)
