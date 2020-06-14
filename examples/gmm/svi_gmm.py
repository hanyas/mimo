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

# plt.figure()
# plt.plot(data[:, 0], data[:, 1], 'kx')
# plt.title('data')

nb_models = 25

gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.mean(data, axis=0), kappa=0.01, psi=np.eye(2), nu=3)
components_prior = distributions.NormalInverseWishart(**components_hypparams)

gmm = mixtures.BayesianMixtureOfGaussians(gating=distributions.CategoricalWithDirichlet(gating_prior),
                                          components=[distributions.GaussianWithNormalInverseWishart(components_prior)
                                                      for _ in range(nb_models)])

gmm.add_data(data)

allscores = []
allmodels = []
for superitr in range(3):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for _ in progprint_xrange(25):
        gmm.resample()

    print('Stochastic Mean Field')
    gmm.resample()
    minibatchsize = 128
    prob = minibatchsize / float(n_samples)
    for _ in progprint_xrange(100):
        minibatch = npr.permutation(n_samples)[:minibatchsize]
        gmm.meanfield_sgdstep(obs=data[minibatch, :], prob=prob, stepsize=5e-4)

    allscores.append(gmm.meanfield_update())
    allmodels.append(copy.deepcopy(gmm))

models_and_scores = sorted([(m, s) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)

plt.figure()
plt.title('best model')
models_and_scores[0][0].plot()
plt.show()
