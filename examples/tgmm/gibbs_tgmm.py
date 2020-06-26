import numpy as np
import numpy.random as npr

from scipy import stats

from matplotlib import pyplot as plt

from mimo import distributions
from mimo import mixtures

from mimo.util.text import progprint_xrange

# npr.seed(1337)

gating = distributions.Categorical(K=2)

lmbda = stats.wishart(3, np.eye(2)).rvs()
ensemble = distributions.TiedGaussiansWithPrecision(mus=[np.array([1., 1.]),
                                                         np.array([-1., -1.])],
                                                    lmbda=lmbda)

gmm = mixtures.MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

obs = [gmm.rvs(100)[0] for _ in range(5)]
gmm.plot(obs)

gating_hypparams = dict(K=2, alphas=np.ones((2, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

ensemble_hypparams = dict(mus=[np.zeros((2, )) for _ in range(2)],
                          kappas=[0.01 for _ in range(2)],
                          psi=np.eye(2), nu=3)
ensemble_prior = distributions.TiedNormalWisharts(**ensemble_hypparams)

model = mixtures.BayesianMixtureOfTiedGaussians(gating=distributions.CategoricalWithDirichlet(gating_prior),
                                                ensemble=distributions.TiedGaussiansWithNormalWishart(ensemble_prior))

model.add_data(obs)

print('Gibbs Sampling')
for _ in progprint_xrange(1000):
    model.resample()

plt.figure()
model.plot(obs)
