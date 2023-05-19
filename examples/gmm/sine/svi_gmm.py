import copy
import operator

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from mimo.distributions import StackedNormalWisharts
from mimo.distributions import StackedGaussiansWithNormalWisharts

from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import BayesianMixtureOfGaussians

from matplotlib import pyplot as plt

npr.seed(1337)

nb_samples = 2500

data = np.zeros((nb_samples, 2))
step = 14. * np.pi / nb_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = x + npr.normal(0, 0.1)
    data[i, 1] = 3. * (np.sin(x) + npr.normal(0, .1))

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.title('data')

nb_models = 25

gating_prior = Dirichlet(dim=nb_models, alphas=np.ones((nb_models, )))

gating = CategoricalWithDirichlet(dim=nb_models, prior=gating_prior)

mus = np.zeros((nb_models, 2))
kappas = 0.01 * np.ones((nb_models,))
psis = np.stack(nb_models * [np.eye(2)])
nus = 3. * np.ones((nb_models,)) + 1e-8

components_prior = StackedNormalWisharts(size=nb_models, dim=2,
                                         mus=mus, kappas=kappas,
                                         psis=psis, nus=nus)

components = StackedGaussiansWithNormalWisharts(size=nb_models, dim=2,
                                                prior=components_prior)

model = BayesianMixtureOfGaussians(gating=gating, components=components)

allvlbs = []
allmodels = []
for superitr in range(5):
    model.resample(data, maxiter=25)
    vlb = model.meanfield_stochastic_descent(obs=data, maxiter=500,
                                             step_size=1e-1, batch_size=256)

    allvlbs.append(vlb)
    allmodels.append(copy.deepcopy(model))

scores = np.array([vlb[-1] for vlb in allvlbs])
best = allmodels[np.argmin(sc.stats.rankdata(-1. * scores))]

plt.figure()
plt.title('Best Model')
best.plot(data)
plt.show()
