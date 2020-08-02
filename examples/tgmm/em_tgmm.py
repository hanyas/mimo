import numpy as np
import numpy.random as npr

from scipy import stats

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import TiedGaussiansWithPrecision
from mimo.mixtures import MixtureOfTiedGaussians

npr.seed(1337)

gating = Categorical(K=2)

lmbda = stats.wishart(3, np.eye(2)).rvs()
ensemble = TiedGaussiansWithPrecision(mus=[np.array([1., 1.]),
                                           np.array([-1., -1.])],
                                      lmbda=lmbda)

gmm = MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

obs = [gmm.rvs(100)[0] for _ in range(5)]
gmm.plot(obs)

gating = Categorical(K=2)
ensemble = TiedGaussiansWithPrecision(mus=[npr.randn(2,),
                                           npr.randn(2,)],
                                      lmbda=np.eye(2))

model = MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

model.max_likelihood(obs, maxiter=500)

plt.figure()
model.plot(obs)
