import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions
from mimo import mixtures

from mimo.util.text import progprint_xrange

import operator

npr.seed(1337)

n_samples = 2500

data = np.zeros((n_samples, 2))
step = 14. * np.pi / n_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = x + npr.normal(0, 0.1)
    data[i, 1] = 3. * (np.sin(x) + npr.normal(0, .1))

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.title('data')

nb_models = 25

gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.zeros((2, )), kappas=0.001 * np.ones((2, )),
                            alphas=9 * np.ones((2, )), betas=1. * np.ones((2, )))
components_prior = distributions.NormalInverseGamma(**components_hypparams)

gmm = mixtures.BayesianMixtureOfGaussians(gating=distributions.CategoricalWithDirichlet(gating_prior),
                                          components=[distributions.GaussianWithNormalInverseGamma(components_prior)
                                                      for _ in range(nb_models)])

gmm.add_data(data)

allscores = []
allmodels = []
for superitr in range(3):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for _ in progprint_xrange(25):
        gmm.resample()

    # mean field to lock onto a mode
    print('Mean Field')
    gmm.resample()  # sample once to initialize posterior
    scores = [gmm.meanfield_update() for _ in progprint_xrange(250)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

plt.figure()
plt.title('model vlb scores vs iteration')
for scores in allscores:
    plt.plot(scores)

models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)

plt.figure()
plt.title('best model')
gmm.plot()
plt.show()
