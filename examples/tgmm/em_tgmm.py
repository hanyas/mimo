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

gating = distributions.Categorical(K=2)
ensemble = distributions.TiedGaussiansWithPrecision(mus=[npr.randn(2, ),
                                                         npr.randn(2,)],
                                                    lmbda=np.eye(2))

model = mixtures.MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

print('Expecation Maximization')
for _ in progprint_xrange(500):
    model.max_likelihood(obs)

plt.figure()
model.plot(obs)
