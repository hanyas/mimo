import numpy as np
import numpy.random as npr

from scipy import stats

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import TiedGaussiansWithPrecision
from mimo.distributions import TiedNormalWisharts
from mimo.distributions import TiedGaussiansWithNormalWishart

from mimo.mixtures import MixtureOfTiedGaussians
from mimo.mixtures import BayesianMixtureOfTiedGaussians

from mimo.util.text import progprint_xrange

# npr.seed(1337)

gating = Categorical(K=2)

lmbda = stats.wishart(3, np.eye(2)).rvs()
ensemble = TiedGaussiansWithPrecision(mus=[np.array([1., 1.]),
                                           np.array([-1., -1.])],
                                      lmbda=lmbda)

gmm = MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

obs = [gmm.rvs(100)[0] for _ in range(5)]
gmm.plot(obs)

gating_hypparams = dict(K=2, alphas=10 * np.ones((2, )))
gating_prior = Dirichlet(**gating_hypparams)

ensemble_hypparams = dict(mus=[np.zeros((2, )) for _ in range(2)],
                          kappas=[0.01 for _ in range(2)],
                          psi=np.eye(2), nu=3)
ensemble_prior = TiedNormalWisharts(**ensemble_hypparams)

model = BayesianMixtureOfTiedGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                       ensemble=TiedGaussiansWithNormalWishart(ensemble_prior))

model.add_data(obs)

vlb = []

model.resample()
print('Variational Inference')
for _ in progprint_xrange(1000):
    vlb.append(model.meanfield_update())

plt.figure()
model.plot(obs)

plt.figure()
plt.plot(vlb)
