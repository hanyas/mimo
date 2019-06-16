import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange

import operator


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

# gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, )))
# gating_prior = distributions.StickBreaking(**gating_hypparams)

components_hypparams = dict(mu=np.mean(data, axis=0), kappa=0.01, psi=np.eye(2), nu=5)
components_prior = distributions.NormalInverseWishart(**components_hypparams)

gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                     components=[distributions.BayesianGaussian(components_prior) for _ in range(nb_models)])

# gmm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
#                      components=[distributions.BayesianGaussian(components_prior) for _ in range(nb_models)])

gmm.add_data(data)

allscores = []
allmodels = []
for superitr in range(5):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for _ in progprint_xrange(100):
        gmm.resample_model()

    print('Stochastic Mean Field')
    minibatchsize = 100
    prob = minibatchsize / float(n_samples)
    for _ in progprint_xrange(150):
        minibatch = npr.permutation(n_samples)[:minibatchsize]
        gmm.meanfield_sgdstep(minibatch=data[minibatch], prob=prob, stepsize=1e-3)

    allscores.append(gmm.meanfield_coordinate_descent_step())
    allmodels.append(copy.deepcopy(gmm))

models_and_scores = sorted([(m, s) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)

plt.figure()
models_and_scores[0][0].plot()
plt.title('best model')
plt.show()
